import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import ObservedHistory, build_normalized_graph, load_interaction_stream, merge_edges, split_prediction_window
from .diagnostics import assign_user_groups, item_cluster_centroids
from .evaluation import evaluate
from .models import build_model


def _checkpoint(path, stream, expected_stage, reference=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"config", "stage", "method", "backbone", "model_state_dict", "graph_edges",
                "data_fingerprint", "num_users", "num_items", "checkpoint_kind"}
    if not required.issubset(payload):
        raise ValueError("User-group evaluation requires complete UGER-format stage checkpoints.")
    if payload["stage"] != expected_stage or payload["checkpoint_kind"] != "continual":
        raise ValueError("The checkpoint stage or checkpoint kind does not match the requested evaluation.")
    if payload["data_fingerprint"] != stream.fingerprint:
        raise ValueError("The checkpoint and interaction file have different fingerprints.")
    if payload["num_users"] != stream.num_users or payload["num_items"] != stream.num_items:
        raise ValueError("The checkpoint catalogue does not match the interaction file.")
    if reference is not None:
        if payload["backbone"] != reference["backbone"]:
            raise ValueError("Shared user cohorts require the same backbone across runs.")
        keys = ("seed", "initial_fraction", "num_snapshots", "validation_fraction", "split_mode", "embedding_dim", "num_layers")
        for key in keys:
            if payload["config"].get(key) != reference["config"].get(key):
                raise ValueError(f"The checkpoint configuration differs for {key}.")
    return payload


def _model(payload, device):
    config = payload["config"]
    extra = {}
    if payload["backbone"].lower().replace("-", "") in {"mgccf", "multigccf"}:
        extra = {"neighbor_samples": config.get("mgccf_neighbors"),
                 "similarity_degree": config.get("similarity_degree", 10)}
    model = build_model(
        payload["backbone"], payload["num_users"], payload["num_items"],
        embedding_dim=config["embedding_dim"], n_layers=config["num_layers"],
        keep_prob=config.get("keep_prob", 1.0), message_dropout=config.get("message_dropout", 0.0),
        **extra,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def _graph(payload, device):
    return build_normalized_graph(payload["graph_edges"], payload["num_users"], payload["num_items"], device)


@torch.no_grad()
def evaluate_user_groups(reference_run, method_runs, data_file, clusters, minimum_history,
                         device="cpu", batch_size=128, max_clustering_iterations=100):
    if clusters < 1 or minimum_history < 1 or batch_size < 1 or max_clustering_iterations < 1:
        raise ValueError("Cluster count, history threshold, batch size, and iteration count must be positive.")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("User-group evaluation supports CPU or CUDA.")
    reference_run = Path(reference_run)
    initial_path = reference_run / "stage_000.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if not isinstance(initial, dict) or not isinstance(initial.get("config"), dict):
        raise ValueError("The reference run does not contain a complete stage-zero checkpoint.")
    config = initial["config"]
    if initial.get("method") != "Fine-tuning":
        raise ValueError("The common reference run must use Fine-tuning.")
    stream = load_interaction_stream(
        data_file, config["initial_fraction"], config["num_snapshots"],
        config["validation_fraction"], config["split_mode"], config.get("delimiter", "\t"),
    )
    initial = _checkpoint(initial_path, stream, 0)
    runs = {"Fine-tuning": reference_run}
    for name, directory in method_runs.items():
        if not name or name == "Fine-tuning":
            raise ValueError("Comparison run labels must be nonempty and distinct from Fine-tuning.")
        runs[name] = Path(directory)
    stages = []
    previous_payload = initial
    for stage in range(1, len(stream.snapshots) - 1):
        after_payload = _checkpoint(reference_run / f"stage_{stage:03d}.pt", stream, stage, initial)
        if after_payload["method"] != "Fine-tuning":
            raise ValueError("Every reference checkpoint must use Fine-tuning.")
        expected = set(map(tuple, stream.snapshots[stage]))
        if set(map(tuple, after_payload["graph_edges"])) != expected:
            raise ValueError("Fine-tuning reference checkpoints must use the incoming-only training graph.")
        previous_history = ObservedHistory(merge_edges(*stream.snapshots[:stage]))
        current_history = ObservedHistory(merge_edges(*stream.snapshots[:stage + 1]))
        previous_model, current_model = _model(previous_payload, device), _model(after_payload, device)
        current_graph = _graph(after_payload, device)
        previous_users, _ = previous_model.encode(current_graph)
        current_users, current_items = current_model.encode(current_graph)
        observed_items = torch.as_tensor(current_history.eligible_items, dtype=torch.long, device=device)
        if not observed_items.numel():
            raise ValueError("User grouping requires observed training items.")
        generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 4099 * stage)
        centroids = item_cluster_centroids(current_items[observed_items], clusters, generator,
                                          max_iter=max_clustering_iterations)
        groups = assign_user_groups(previous_users, current_users, centroids, previous_history,
                                    current_history, minimum_history)
        validation_edges, test_edges = split_prediction_window(stream.snapshots[stage + 1])
        results = {}
        for name, directory in runs.items():
            payload = after_payload if name == "Fine-tuning" else _checkpoint(
                directory / f"stage_{stage:03d}.pt", stream, stage, initial
            )
            model = current_model if name == "Fine-tuning" else _model(payload, device)
            graph = current_graph if name == "Fine-tuning" else _graph(payload, device)
            results[name] = {
                group: {
                    "validation": evaluate(model, graph, validation_edges, current_history, (10,), batch_size, users=groups[group]),
                    "test": evaluate(model, graph, test_edges, current_history, (10,), batch_size, users=groups[group]),
                }
                for group in ("stable", "dynamic")
            }
        stages.append({
            "trained_through_stage": stage, "evaluated_on_stage": stage + 1,
            "clusters_used": len(centroids), "groups": groups, "methods": results,
        })
        previous_payload = after_payload
    means = {}
    for name in runs:
        means[name] = {}
        for group in ("stable", "dynamic"):
            values = [record["methods"][name][group]["test"]["metrics"]["NDCG@10"]
                      for record in stages if record["methods"][name][group]["test"]["evaluated_users"]]
            means[name][group] = float(np.mean(values)) if values else None
    return {
        "seed": int(config["seed"]), "backbone": initial["backbone"],
        "clusters_requested": int(clusters), "minimum_history": int(minimum_history),
        "metric_scale": "fraction", "group_stage_mean_ndcg10": means,
        "skipped_trained_stages": [{"stage": 0, "reason": "No pre-pretraining checkpoint is available for a shared drift measurement."}],
        "stages": stages,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate shared training-only preference-drift user groups")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--method-run", action="append", default=[], metavar="NAME=DIRECTORY")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--minimum-history", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-clustering-iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    methods = {}
    for specification in args.method_run:
        if "=" not in specification:
            parser.error("--method-run must have the form NAME=DIRECTORY")
        name, directory = specification.split("=", 1)
        if not name or not directory or name in methods:
            parser.error("Comparison run labels must be unique and both fields must be nonempty")
        methods[name] = Path(directory)
    args.method_runs = methods
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError("The user-group output file already exists.")
    result = evaluate_user_groups(
        args.reference_run, args.method_runs, args.data_file, args.clusters, args.minimum_history,
        args.device, args.batch_size, args.max_clustering_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return args.output


if __name__ == "__main__":
    main()
