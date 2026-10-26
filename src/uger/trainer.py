import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import ObservedHistory, build_normalized_graph, load_interaction_stream, merge_edges, sample_recommendation_batch, split_prediction_window
from .evaluation import evaluate
from .models import build_model
from .replay import select_replay_buffer
from .stability import UGERStability


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(args):
    if args.gpu is not None:
        if args.gpu < 0:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
        return torch.device(f"cuda:{args.gpu}")
    if args.device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use CPU or CUDA for sparse graph propagation")
    return device


def synchronized_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def json_config(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def subset_batch(batch, mask):
    return {key: value[mask] for key, value in batch.items()}


def paired_batches(incoming, buffer, batch_size, rng):
    empty = np.empty((0, 3), dtype=np.int64)
    if not len(incoming) or not len(buffer):
        source = incoming if len(incoming) else buffer
        order = rng.permutation(len(source))
        for start in range(0, len(order), batch_size):
            batch = source[order[start:start + batch_size]]
            yield (batch, empty) if len(incoming) else (empty, batch)
        return
    if batch_size < 2:
        raise ValueError("Paired incoming and replay batches require a batch size of at least two")
    incoming_size = min(batch_size - 1, max(1, round(batch_size * len(incoming) / (len(incoming) + len(buffer)))))
    replay_size = batch_size - incoming_size
    incoming_order = rng.permutation(len(incoming))
    replay_order = rng.permutation(len(buffer))
    incoming_steps = (len(incoming) + incoming_size - 1) // incoming_size
    replay_steps = (len(buffer) + replay_size - 1) // replay_size
    for step in range(max(incoming_steps, replay_steps)):
        incoming_step, replay_step = step % incoming_steps, step % replay_steps
        if step and incoming_step == 0:
            incoming_order = rng.permutation(len(incoming))
        if step and replay_step == 0:
            replay_order = rng.permutation(len(buffer))
        new_indices = incoming_order[incoming_step * incoming_size:(incoming_step + 1) * incoming_size]
        replay_indices = replay_order[replay_step * replay_size:(replay_step + 1) * replay_size]
        yield incoming[new_indices], buffer[replay_indices]


class UGERTrainer:
    def __init__(self, args, stream, directory, device):
        self.args = args
        self.stream = stream
        self.directory = Path(directory)
        self.device = device
        self.rng = np.random.default_rng(args.seed)
        self.selection_rng = np.random.default_rng(args.seed + 1009)
        self.generator = torch.Generator(device=device).manual_seed(args.seed + 2017)
        snapshots = getattr(stream, "snapshots", [[], []])
        self.buffer_capacity = args.buffer_size if args.buffer_size is not None else len(snapshots[1])
        self.buffer_schedule = {}
        self.fixed_schedule = {}
        if args.replay_schedule:
            with np.load(args.replay_schedule, allow_pickle=False) as schedule:
                if str(schedule["data_fingerprint"].item()) != stream.fingerprint:
                    raise ValueError("Replay schedule and interaction stream do not match")
                if int(schedule["buffer_capacity"].item()) != self.buffer_capacity:
                    raise ValueError("Replay schedule and buffer capacity do not match")
                self.fixed_schedule = {key: schedule[key].copy() for key in schedule.files if key.startswith("stage_")}
        self.model = self.make_model()
        self.optimizer = self.make_optimizer()
        self.stability = UGERStability(
            epsilon=args.epsilon, proposal_size=args.proposal_size,
            clean_quota=args.clean_quota, probe_quota=args.probe_quota,
            nu=args.normalization_floor, fallback_scale=args.fallback_scale,
            refine=not args.no_refinement,
            candidate_policy=args.candidate_policy,
            initialization=args.perturbation_init,
            candidate_probe=args.candidate_probe,
            random_scale=args.random_scale,
        )
        self.results = []
        self.elapsed_training_seconds = 0.0
        self.stage_training_seconds = []

    def make_model(self):
        extra = {}
        if self.args.backbone.lower().replace("-", "").replace("_", "") in {"mgccf", "multigccf"}:
            extra = {"neighbor_samples": tuple(self.args.mgccf_neighbors), "similarity_degree": self.args.similarity_degree}
        return build_model(
            self.args.backbone, self.stream.num_users, self.stream.num_items,
            embedding_dim=self.args.embedding_dim, n_layers=self.args.num_layers,
            keep_prob=self.args.keep_prob, message_dropout=self.args.message_dropout,
            **extra,
        ).to(self.device)

    def make_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def graph(self, edges):
        return build_normalized_graph(edges, self.stream.num_users, self.stream.num_items, self.device)

    def train_epoch(self, incoming, buffer, graph, history, consistency):
        self.model.train()
        statistics = {"recommendation_loss": 0.0, "consistency_loss": 0.0, "steps": 0,
                      "training_rows": 0, "skipped_rows": 0, "probe_fallbacks": 0}
        for new_edges, replay_edges in paired_batches(incoming, buffer, self.args.batch_size, self.rng):
            pool = np.concatenate((new_edges, replay_edges), axis=0)
            batch = sample_recommendation_batch(
                pool, history, self.args.num_negatives, self.rng, device=self.device
            )
            statistics["skipped_rows"] += len(pool) - len(batch["users"])
            if not len(batch["users"]):
                continue
            incoming_mask = batch["row_indices"] < len(new_edges)
            masks = self.model.sample_masks(graph, self.generator)
            consistency_loss = next(self.model.parameters()).sum() * 0.0
            if consistency and len(replay_edges) and self.args.consistency_weight > 0:
                replay_batch = {
                    "users": torch.as_tensor(replay_edges[:, 0], dtype=torch.long, device=self.device),
                    "items": torch.as_tensor(replay_edges[:, 1:2], dtype=torch.long, device=self.device),
                    "timestamps": torch.as_tensor(replay_edges[:, 2], dtype=torch.long, device=self.device),
                }
                consistency_loss, diagnostics = self.stability.compute(
                    self.model, subset_batch(batch, incoming_mask), replay_batch,
                    graph, history.user_items, history.eligible_items, self.generator, masks=masks,
                )
                statistics["probe_fallbacks"] += int(diagnostics["fallback_used"])
            recommendation_loss = self.model.recommendation_loss(batch, graph, masks=masks)
            objective = recommendation_loss + self.args.consistency_weight * consistency_loss
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite training objective")
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward()
            for parameter in self.model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Non-finite model gradient")
            self.optimizer.step()
            statistics["recommendation_loss"] += float(recommendation_loss.detach())
            statistics["consistency_loss"] += float(consistency_loss.detach())
            statistics["training_rows"] += len(batch["users"])
            statistics["steps"] += 1
        for key in ("recommendation_loss", "consistency_loss"):
            statistics[key] /= max(statistics["steps"], 1)
        return statistics

    def fit(self, incoming, buffer, graph, history, epochs, stage, consistency=False):
        for epoch in range(1, epochs + 1):
            statistics = self.train_epoch(incoming, buffer, graph, history, consistency)
            if not statistics["steps"] and len(incoming) + len(buffer):
                logging.warning("Stage %s has no legal recommendation comparisons", stage)
            logging.info("stage=%s epoch=%d/%d rec=%.6f con=%.6f steps=%d skipped=%d fallbacks=%d",
                         stage, epoch, epochs, statistics["recommendation_loss"], statistics["consistency_loss"],
                         statistics["steps"], statistics["skipped_rows"], statistics["probe_fallbacks"])

    def pretrain(self):
        empty = np.empty((0, 3), dtype=np.int64)
        history = ObservedHistory(self.stream.initial_train)
        graph = self.graph(self.stream.initial_train)
        best_epoch, best_metric, stale = 1, -1.0, 0
        for epoch in range(1, self.args.pretrain_epochs + 1):
            training_start = synchronized_time(self.device)
            statistics = self.train_epoch(self.stream.initial_train, empty, graph, history, False)
            self.elapsed_training_seconds += synchronized_time(self.device) - training_start
            logging.info("stage=validation epoch=%d rec=%.6f steps=%d", epoch, statistics["recommendation_loss"], statistics["steps"])
            if epoch % self.args.evaluation_interval and epoch != self.args.pretrain_epochs:
                continue
            validation = evaluate(self.model, graph, self.stream.initial_validation, history,
                                  top_k=(10,), batch_size=self.args.evaluation_batch_size)
            metric = validation["metrics"]["Recall@10"]
            if metric > best_metric:
                best_metric, best_epoch, stale = metric, epoch, 0
            else:
                stale += 1
            logging.info("validation epoch=%d recall10=%.6f users=%d", epoch, metric, validation["evaluated_users"])
            if self.args.patience and stale >= self.args.patience:
                break
        seed_everything(self.args.seed)
        self.rng = np.random.default_rng(self.args.seed)
        self.generator.manual_seed(self.args.seed + 2017)
        self.model = self.make_model()
        self.optimizer = self.make_optimizer()
        initial = self.stream.snapshots[0]
        history, graph = ObservedHistory(initial), self.graph(initial)
        training_start = synchronized_time(self.device)
        self.fit(initial, empty, graph, history, best_epoch, stage=0)
        self.elapsed_training_seconds += synchronized_time(self.device) - training_start
        self.save(self.directory / "pretrained.pt", stage=0, buffer=empty,
                  graph_edges=initial, checkpoint_kind="pretrained", selected_epochs=best_epoch)
        return history, graph, initial

    def refresh_buffer(self, pool, graph, history, history_size, stage=0):
        if self.args.method not in {"UGER", "Uniform", "Invdeg", "INFER"}:
            return np.empty((0, 3), dtype=np.int64)
        start = synchronized_time(self.device)
        key = f"stage_{stage:03d}"
        if self.fixed_schedule:
            if key not in self.fixed_schedule:
                raise ValueError(f"Replay schedule is missing {key}")
            edges = np.asarray(self.fixed_schedule[key], dtype=np.int64).reshape(-1, 3)
            if len(edges) != min(self.buffer_capacity, len(pool)):
                raise ValueError("Replay schedule violates the fixed buffer capacity")
            if len(merge_edges(edges)) != len(edges) or not set(map(tuple, edges)).issubset(set(map(tuple, pool))):
                raise ValueError("Replay schedule contains repeated or unavailable interactions")
        else:
            selection = select_replay_buffer(
                self.model, pool, graph, history, self.buffer_capacity, self.args.beta,
                self.args.num_negatives, self.args.selection_batch_size, self.selection_rng,
                method=self.args.method,
            )
            edges = selection.edges
        self.elapsed_training_seconds += synchronized_time(self.device) - start
        self.buffer_schedule[key] = edges.copy()
        logging.info("buffer_size=%d capacity=%d selection_pool=%d history_size=%d", len(edges), self.buffer_capacity, len(pool), history_size)
        return edges

    def rng_state(self):
        return {"python": random.getstate(), "numpy": self.rng.bit_generator.state,
                "selection_numpy": self.selection_rng.bit_generator.state, "torch": torch.get_rng_state(),
                "generator": self.generator.get_state(),
                "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None}

    def save(self, path, stage, buffer, graph_edges, checkpoint_kind="continual", selected_epochs=None):
        save_checkpoint(path, {"format_version": 2, "method": self.args.method,
                              "backbone": self.args.backbone, "config": json_config(self.args),
                              "data_fingerprint": self.stream.fingerprint,
                              "num_users": self.stream.num_users, "num_items": self.stream.num_items,
                              "model_state_dict": self.model.state_dict(),
                              "optimizer_state_dict": self.optimizer.state_dict(),
                              "stage": stage, "buffer": buffer, "graph_edges": graph_edges,
                              "checkpoint_kind": checkpoint_kind, "selected_pretraining_epochs": selected_epochs,
                              "rng_state": self.rng_state(), "results": self.results,
                              "training_seconds": self.elapsed_training_seconds,
                              "stage_training_seconds": self.stage_training_seconds,
                              "buffer_capacity": self.buffer_capacity,
                              "buffer_schedule": self.buffer_schedule,
                              "fixed_replay_schedule": self.fixed_schedule})

    def load(self, path, resume):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("format_version") != 2:
            raise ValueError("The checkpoint does not use the UGER checkpoint format")
        if payload["data_fingerprint"] != self.stream.fingerprint:
            raise ValueError("Checkpoint and input interaction file do not match")
        config = payload["config"]
        required = ["embedding_dim", "num_layers", "initial_fraction", "num_snapshots", "validation_fraction", "split_mode", "seed", "mgccf_neighbors", "similarity_degree"]
        if str(payload["backbone"]).lower() != str(self.args.backbone).lower():
            raise ValueError("Checkpoint backbone does not match the requested backbone")
        if resume:
            if payload["checkpoint_kind"] != "continual" or payload["method"] != self.args.method:
                raise ValueError("Resume requires a continual checkpoint from the same method")
            required += ["buffer_size", "beta", "epsilon", "consistency_weight", "proposal_size", "clean_quota", "probe_quota",
                         "normalization_floor", "fallback_scale", "no_refinement", "learning_rate", "weight_decay",
                         "num_negatives", "batch_size", "incremental_epochs", "keep_prob", "message_dropout",
                         "selection_batch_size", "num_threads", "top_k", "variant", "candidate_policy",
                         "perturbation_init", "candidate_probe", "random_scale"]
            if payload["buffer_capacity"] != self.buffer_capacity:
                raise ValueError("Checkpoint buffer capacity differs from this run")
            scheduled = payload.get("fixed_replay_schedule", {})
            if self.fixed_schedule and (set(self.fixed_schedule) != set(scheduled) or any(
                    not np.array_equal(self.fixed_schedule[key], scheduled[key]) for key in self.fixed_schedule)):
                raise ValueError("The supplied replay schedule differs from the resumed checkpoint")
            self.fixed_schedule = scheduled
        elif payload["checkpoint_kind"] != "pretrained" or payload["stage"] != 0:
            raise ValueError("--pretrained requires a stage-zero pretrained checkpoint")
        for key in required:
            if config[key] != getattr(self.args, key):
                raise ValueError(f"Checkpoint configuration differs for {key}")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = payload["rng_state"]
        random.setstate(state["python"])
        self.rng.bit_generator.state = state["numpy"]
        self.selection_rng.bit_generator.state = state["selection_numpy"]
        torch.set_rng_state(state["torch"].cpu())
        self.generator.set_state(state["generator"].cpu())
        if state["cuda"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        if resume:
            self.results = payload["results"]
            self.elapsed_training_seconds = payload["training_seconds"]
            self.stage_training_seconds = payload["stage_training_seconds"]
            self.buffer_schedule = payload["buffer_schedule"]
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = self.args.learning_rate
                group["weight_decay"] = self.args.weight_decay
        return payload

    def continual(self):
        if self.args.resume:
            payload = self.load(self.args.resume, resume=True)
            stage = payload["stage"]
            observed = merge_edges(*self.stream.snapshots[:stage + 1])
            history = ObservedHistory(observed)
            buffer = np.asarray(payload["buffer"], dtype=np.int64)
            graph_edges = np.asarray(payload["graph_edges"], dtype=np.int64)
            graph = self.graph(graph_edges)
            next_stage = stage + 1
        else:
            if self.args.pretrained:
                self.load(self.args.pretrained, resume=False)
                observed = self.stream.snapshots[0]
                history, graph_edges = ObservedHistory(observed), observed
                graph = self.graph(graph_edges)
            else:
                history, graph, graph_edges = self.pretrain()
                observed = self.stream.snapshots[0]
            buffer = self.refresh_buffer(graph_edges, graph, history, len(observed), stage=0)
            self.save(self.directory / "stage_000.pt", 0, buffer, graph_edges)
            next_stage = 1
        save_json(self.directory / "metrics.json", self.results)
        if next_stage >= len(self.stream.snapshots):
            self.save(self.directory / f"stage_{next_stage - 1:03d}.pt", next_stage - 1, buffer, graph_edges)
        for stage in range(next_stage, len(self.stream.snapshots)):
            incoming = self.stream.snapshots[stage]
            validation_edges, test_edges = split_prediction_window(incoming)
            validation = evaluate(self.model, graph, validation_edges, history, tuple(self.args.top_k), self.args.evaluation_batch_size)
            result = evaluate(self.model, graph, test_edges, history, tuple(self.args.top_k), self.args.evaluation_batch_size)
            result["validation"] = validation
            result.update({"trained_through_stage": stage - 1, "evaluated_on_stage": stage})
            self.results.append(result)
            save_json(self.directory / "metrics.json", self.results)
            logging.info("evaluation_stage=%d metrics=%s users=%d", stage, result["metrics"], result["evaluated_users"])
            observed = merge_edges(observed, incoming)
            history.add(incoming)
            if self.args.method == "Retraining":
                seed_everything(self.args.seed + stage)
                self.model, graph_edges = self.make_model(), observed
                self.optimizer = self.make_optimizer()
                training_incoming, training_buffer = observed, np.empty((0, 3), dtype=np.int64)
            else:
                training_incoming, training_buffer = incoming, buffer
                graph_edges = merge_edges(incoming, buffer)
            graph = self.graph(graph_edges)
            stage_start = self.elapsed_training_seconds
            start = synchronized_time(self.device)
            self.fit(training_incoming, training_buffer, graph, history, self.args.incremental_epochs,
                     stage, consistency=self.args.method == "UGER")
            self.elapsed_training_seconds += synchronized_time(self.device) - start
            buffer = self.refresh_buffer(graph_edges, graph, history, len(observed), stage=stage)
            self.stage_training_seconds.append(self.elapsed_training_seconds - stage_start)
            if self.args.save_every_stage or stage == len(self.stream.snapshots) - 1:
                self.save(self.directory / f"stage_{stage:03d}.pt", stage, buffer, graph_edges)
        summary = {"method": self.args.method, "backbone": self.args.backbone,
                   "variant": self.args.variant,
                   "stage_mean": {}, "evaluated_stages": len(self.results),
                   "training_seconds_including_buffer_selection": self.elapsed_training_seconds,
                   "incremental_stage_seconds": self.stage_training_seconds,
                   "mean_incremental_stage_seconds": float(np.mean(self.stage_training_seconds)) if self.stage_training_seconds else 0.0,
                   "timing_excludes_evaluation_and_checkpoint_io": True}
        available = [result for result in self.results if result["evaluated_users"]]
        for key in self.results[0]["metrics"] if self.results else []:
            summary["stage_mean"][key] = float(np.mean([result["metrics"][key] for result in available])) if available else 0.0
        save_json(self.directory / "summary.json", summary)
        np.savez_compressed(self.directory / "buffers.npz", data_fingerprint=np.asarray(self.stream.fingerprint),
                            buffer_capacity=np.asarray(self.buffer_capacity), **self.buffer_schedule)


def run(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.num_threads)
    device = resolve_device(args)
    aliases = {"lgn": "LightGCN", "lightgcn": "LightGCN", "ngcf": "NGCF", "mgccf": "MGCCF", "multigccf": "MGCCF", "mf": "MF", "matrixfactorization": "MF"}
    args.backbone = aliases.get(args.backbone.lower().replace("-", "").replace("_", ""), args.backbone)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    name = args.run_name or f"{args.method}_{args.backbone}_{args.dataset}_seed{args.seed}_{stamp}"
    directory = args.output_dir / name
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(directory / "training.log"), logging.StreamHandler()], force=True)
    logging.info("method=%s backbone=%s device=%s", args.method, args.backbone, device)
    stream = load_interaction_stream(args.data_file, args.initial_fraction, args.num_snapshots,
                                     args.validation_fraction, args.split_mode, args.delimiter)
    if args.buffer_size is None:
        args.buffer_size = len(stream.snapshots[1])
    configuration = json_config(args)
    configuration.update({"data_fingerprint": stream.fingerprint, "snapshot_sizes": [len(edges) for edges in stream.snapshots],
                          "num_users": stream.num_users, "num_items": stream.num_items,
                          "torch_version": torch.__version__})
    save_json(directory / "config.json", configuration)
    save_json(directory / "id_mapping.json", {"users": stream.user_mapping, "items": stream.item_mapping})
    trainer = UGERTrainer(args, stream, directory, device)
    if args.mode == "pretrain":
        if args.resume or args.pretrained:
            raise ValueError("Pretraining mode does not accept an existing checkpoint")
        trainer.pretrain()
    else:
        trainer.continual()
    logging.info("run_directory=%s", directory)
    return directory
