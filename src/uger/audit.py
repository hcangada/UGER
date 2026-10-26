from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import ObservedHistory, build_normalized_graph, load_interaction_stream, merge_edges, sample_recommendation_batch
from .diagnostics import evaluate_competitor_coverage, evaluate_replay_turnover
from .models import build_model
from .stability import UGERStability, build_ranked_candidates


@dataclass
class StageAuditCache:
    replay_edges: np.ndarray
    proposal_pools: list[np.ndarray]
    fixed_candidates: list[np.ndarray]
    history: ObservedHistory


def _generator(seed, device):
    return torch.Generator(device=device).manual_seed(int(seed))


def _replay_batch(edges, device):
    return {
        "users": torch.as_tensor(edges[:, 0], dtype=torch.long, device=device),
        "items": torch.as_tensor(edges[:, 1:2], dtype=torch.long, device=device),
    }


@torch.no_grad()
def cache_stage_supports(model, graph, replay_edges, history, stability, generator, masks=None):
    edges = np.asarray(replay_edges, dtype=np.int64).reshape(-1, 3).copy()
    if not len(edges):
        raise ValueError("A replay diagnostic requires nonempty replay interactions.")
    device = next(model.parameters()).device
    replay = _replay_batch(edges, device)
    if masks is None:
        masks = model.sample_masks(graph, generator)
    pools = stability._proposals(replay["users"], replay["items"][:, 0], history.eligible_items, history.user_items, generator, device)
    packed, _ = stability._pack(pools, replay["items"][:, 0])
    scores = model(replay["users"], packed, graph, masks=masks)
    size = stability.clean_quota + stability.probe_quota + 1
    fixed = [build_ranked_candidates(pool, scores[index, :len(pool)], int(edge[1]), size).cpu().numpy().copy()
             for index, (edge, pool) in enumerate(zip(edges, pools))]
    return StageAuditCache(edges, [pool.cpu().numpy().copy() for pool in pools], fixed, copy.deepcopy(history))


def _clone_training_state(model, optimizer):
    cloned_model, cloned_optimizer = copy.deepcopy((model, optimizer))
    model_parameters = list(cloned_model.parameters())
    optimizer_parameters = [parameter for group in cloned_optimizer.param_groups for parameter in group["params"]]
    if {id(parameter) for parameter in model_parameters if parameter.requires_grad} != {id(parameter) for parameter in optimizer_parameters if parameter.requires_grad}:
        raise ValueError("The optimizer must own all trainable model parameters.")
    return cloned_model, cloned_optimizer


def _optimizer_step(model, optimizer, objective):
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError("The diagnostic objective is not finite.")
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
        raise FloatingPointError("The diagnostic gradient is not finite.")
    optimizer.step()


def _options(stability, policy):
    return {
        "epsilon": stability.epsilon,
        "proposal_size": stability.proposal_size,
        "clean_quota": stability.clean_quota,
        "probe_quota": stability.probe_quota,
        "nu": stability.nu,
        "fallback_scale": stability.fallback_scale,
        "refine": stability.refine,
        "candidate_policy": {"Fixed": "enhanced", "Clean": "clean", "Probe": "probe", "UGER": "enhanced"}[policy],
        "initialization": "incoming",
        "candidate_probe": "incoming",
        "random_scale": stability.random_scale,
    }


def audit_candidate_policies(model, optimizer, graph, history, mining_edges, reference_edges, cache, stability,
                             consistency_weight=1.0, num_negatives=4, k=10, seed=2021, masks=None,
                             policies=("Fixed", "Clean", "Probe", "UGER")):
    mining_edges = np.asarray(mining_edges, dtype=np.int64).reshape(-1, 3)
    reference_edges = np.asarray(reference_edges, dtype=np.int64).reshape(-1, 3)
    if not len(mining_edges) or not len(reference_edges) or not len(cache.replay_edges):
        raise ValueError("Mining, reference, and replay batches must be nonempty.")
    if set(map(tuple, mining_edges)).intersection(map(tuple, reference_edges)):
        raise ValueError("Mining and reference incoming batches must be disjoint.")
    if history.user_items != cache.history.user_items:
        raise ValueError("The stage-cached eligibility history must remain fixed.")
    if consistency_weight < 0 or not np.isfinite(consistency_weight) or num_negatives < 1 or k < 1:
        raise ValueError("Audit weights, recommendation negatives, and ranking cutoffs are invalid.")
    policies = tuple(policies)
    if not policies or len(set(policies)) != len(policies) or any(policy not in {"Fixed", "Clean", "Probe", "UGER"} for policy in policies):
        raise ValueError("Choose distinct Fixed, Clean, Probe, and UGER audit policies.")
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    paired_edges = np.concatenate((mining_edges, cache.replay_edges))
    paired = sample_recommendation_batch(paired_edges, history, num_negatives, rng, device)
    incoming = {name: values[paired["row_indices"] < len(mining_edges)] for name, values in paired.items()}
    reference_batch = sample_recommendation_batch(reference_edges, history, num_negatives, rng, device)
    if not len(incoming["users"]) or not len(reference_batch["users"]):
        raise ValueError("Both incoming batches require legal recommendation comparisons.")
    if masks is None:
        masks = model.sample_masks(graph, _generator(seed + 1, device))
    reference_model, reference_optimizer = _clone_training_state(model, optimizer)
    reference_objective = reference_model.recommendation_loss(reference_batch, graph, masks=masks)
    _optimizer_step(reference_model, reference_optimizer, reference_objective)
    replay = _replay_batch(cache.replay_edges, device)
    reports = {}
    for policy in policies:
        working_model, working_optimizer = _clone_training_state(model, optimizer)
        module = UGERStability(**_options(stability, policy))
        fixed = cache.fixed_candidates if policy == "Fixed" else None
        consistency, details = module.compute(
            working_model, incoming, replay, graph, history.user_items, history.eligible_items,
            _generator(seed + 2, device), masks=masks, proposal_pools=cache.proposal_pools,
            candidate_sets=fixed,
        )
        supports = [row[mask].cpu().numpy() for row, mask in zip(details["candidate_items"], details["candidate_mask"])]
        coverage = evaluate_competitor_coverage(model, reference_model, graph, cache.replay_edges,
                                                cache.proposal_pools, supports, k=k, masks=masks)
        recommendation = working_model.recommendation_loss(paired, graph, masks=masks)
        objective = recommendation + consistency_weight * consistency
        _optimizer_step(working_model, working_optimizer, objective)
        turnover = evaluate_replay_turnover(model, working_model, graph, cache.replay_edges,
                                             cache.history, k=k, masks=masks)
        reports[policy] = {
            "coverage": coverage,
            "turnover": turnover,
            "recommendation_loss": float(recommendation.detach().cpu()),
            "consistency_loss": float(consistency.detach().cpu()),
            "candidate_sets": [support.tolist() for support in supports],
            "probe_norm": details["probe_norm"],
            "selected_perturbation_norm": details["perturbation_norm"],
        }
    return {
        "seed": int(seed),
        "ranking_cutoff": int(k),
        "coverage_ranking_domain": "shared_stage_cached_proposal_pool",
        "turnover_ranking_domain": "full_catalogue_with_anchor_and_without_other_history_positives",
        "incoming_reference_update": "recommendation_only_with_cloned_optimizer_state",
        "incoming_batches_disjoint": True,
        "mining_interactions": len(mining_edges),
        "reference_interactions": len(reference_edges),
        "replay_interactions": len(cache.replay_edges),
        "policies": reports,
    }


def _load_stage(path, data_file, delimiter, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 2 or payload.get("checkpoint_kind") != "continual":
        raise ValueError("The diagnostic requires a completed-stage UGER format-two checkpoint.")
    config = payload["config"]
    stream = load_interaction_stream(
        data_file, config["initial_fraction"], config["num_snapshots"],
        config["validation_fraction"], config["split_mode"], delimiter,
    )
    if payload["data_fingerprint"] != stream.fingerprint:
        raise ValueError("Checkpoint and input interaction file do not match.")
    if payload["num_users"] != stream.num_users or payload["num_items"] != stream.num_items:
        raise ValueError("Checkpoint dimensions do not match the input catalogue.")
    stage = int(payload["stage"]) + 1
    if stage >= len(stream.snapshots):
        raise ValueError("The checkpoint has no subsequent incoming training stage to audit.")
    extra = {}
    if payload["backbone"].lower() == "mgccf":
        extra = {"neighbor_samples": tuple(config["mgccf_neighbors"]), "similarity_degree": config["similarity_degree"]}
    model = build_model(
        payload["backbone"], stream.num_users, stream.num_items,
        embedding_dim=config["embedding_dim"], n_layers=config["num_layers"],
        keep_prob=config["keep_prob"], message_dropout=config["message_dropout"], **extra,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    buffer = np.asarray(payload["buffer"], dtype=np.int64).reshape(-1, 3)
    incoming = stream.snapshots[stage]
    history = ObservedHistory(merge_edges(*stream.snapshots[:stage + 1]))
    graph = build_normalized_graph(merge_edges(incoming, buffer), stream.num_users, stream.num_items, device)
    stability = UGERStability(
        epsilon=config["epsilon"], proposal_size=config["proposal_size"],
        clean_quota=config["clean_quota"], probe_quota=config["probe_quota"],
        nu=config["normalization_floor"], fallback_scale=config["fallback_scale"],
        refine=True, random_scale=config["random_scale"],
    )
    return payload, stream, stage, model, optimizer, graph, history, incoming, buffer, stability


def _draw_disjoint_batches(incoming, size, rng):
    size = min(int(size), len(incoming) // 2)
    if size < 1:
        raise ValueError("At least two incoming interactions are required for disjoint audit batches.")
    order = rng.permutation(len(incoming))[:2 * size]
    return incoming[order[:size]], incoming[order[size:]]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Controlled candidate coverage and full-catalogue turnover diagnostics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delimiter", default="\t")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--incoming-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--policies", nargs="+", choices=["Fixed", "Clean", "Probe", "UGER"], default=["Fixed", "Clean", "Probe", "UGER"])
    args = parser.parse_args(argv)
    if min(args.incoming_size, args.replay_size, args.top_k) < 1 or args.warmup_steps < 0:
        parser.error("Batch sizes and ranking cutoffs must be positive; warmup steps must be nonnegative")
    if args.output.exists():
        parser.error("The diagnostic output file already exists")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("Use CPU or CUDA for graph diagnostics")
    torch.set_num_threads(1)
    payload, stream, stage, model, optimizer, graph, history, incoming, buffer, stability = _load_stage(args.checkpoint, args.data_file, args.delimiter, device)
    if not len(buffer):
        parser.error("The checkpoint replay buffer is empty")
    rng = np.random.default_rng(args.seed)
    replay = buffer[rng.permutation(len(buffer))[:min(args.replay_size, len(buffer))]]
    cache = cache_stage_supports(model, graph, replay, history, stability, _generator(args.seed + 11, device))
    for step in range(args.warmup_steps):
        count = min(args.incoming_size, len(incoming))
        new_edges = incoming[rng.permutation(len(incoming))[:count]]
        paired = sample_recommendation_batch(np.concatenate((new_edges, replay)), history, payload["config"]["num_negatives"], rng, device)
        incoming_batch = {name: value[paired["row_indices"] < len(new_edges)] for name, value in paired.items()}
        if not len(paired["users"]):
            raise ValueError("The warmup batch has no legal recommendation comparisons.")
        generator = _generator(args.seed + 101 + step, device)
        masks = model.sample_masks(graph, generator)
        consistency, _ = stability.compute(model, incoming_batch, _replay_batch(replay, device), graph,
                                           history.user_items, history.eligible_items, generator,
                                           masks=masks, proposal_pools=cache.proposal_pools)
        objective = model.recommendation_loss(paired, graph, masks=masks) + payload["config"]["consistency_weight"] * consistency
        _optimizer_step(model, optimizer, objective)
    mining, reference = _draw_disjoint_batches(incoming, args.incoming_size, rng)
    report = audit_candidate_policies(
        model, optimizer, graph, history, mining, reference, cache, stability,
        consistency_weight=payload["config"]["consistency_weight"],
        num_negatives=payload["config"]["num_negatives"], k=args.top_k,
        seed=args.seed, policies=args.policies,
    )
    report.update({"training_stage": stage, "stage_start_checkpoint": int(payload["stage"]),
                   "warmup_steps": args.warmup_steps, "backbone": payload["backbone"],
                   "num_catalogue_items": stream.num_items})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
