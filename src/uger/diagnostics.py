from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def item_cluster_centroids(item_embeddings, num_clusters, generator=None, max_iter=100, tolerance=1e-6):
    if item_embeddings.ndim != 2 or not len(item_embeddings):
        raise ValueError("Item embeddings must be a nonempty matrix.")
    if not item_embeddings.is_floating_point() or not torch.isfinite(item_embeddings).all():
        raise ValueError("Item embeddings must be finite floating-point values.")
    if num_clusters < 1 or max_iter < 1 or tolerance < 0:
        raise ValueError("Clustering parameters must be positive, with nonnegative tolerance.")
    count = min(int(num_clusters), len(item_embeddings))
    sampling_device = item_embeddings.device if generator is None else generator.device
    selected = torch.randperm(len(item_embeddings), generator=generator, device=sampling_device)[:count]
    centroids = item_embeddings[selected.to(item_embeddings.device)].clone()
    for _ in range(max_iter):
        labels = torch.cdist(item_embeddings, centroids).argmin(dim=1)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, labels, item_embeddings)
        sizes = torch.bincount(labels, minlength=count).to(item_embeddings.dtype)
        updated = torch.where(sizes[:, None] > 0, sums / sizes.clamp_min(1)[:, None], centroids)
        if torch.allclose(updated, centroids, rtol=0, atol=tolerance):
            centroids = updated
            break
        centroids = updated
    return centroids.detach()


@torch.no_grad()
def preference_shift_scores(before_user_embeddings, after_user_embeddings, centroids):
    if before_user_embeddings.shape != after_user_embeddings.shape or before_user_embeddings.ndim != 2:
        raise ValueError("Pre-update and post-update user embeddings must have identical matrix shapes.")
    if centroids.ndim != 2 or not len(centroids) or centroids.shape[1] != before_user_embeddings.shape[1]:
        raise ValueError("Centroids must match the user embedding dimension.")
    before = torch.log_softmax(before_user_embeddings @ centroids.T, dim=1)
    after = torch.log_softmax(after_user_embeddings @ centroids.T, dim=1)
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise FloatingPointError("Cluster-affinity distributions must be finite.")
    mixture = torch.logaddexp(before, after) - math.log(2)
    values = 0.5 * ((before.exp() * (before - mixture)).sum(dim=1)
                    + (after.exp() * (after - mixture)).sum(dim=1))
    return values.clamp_min(0)


def assign_user_groups(before_user_embeddings, after_user_embeddings, centroids,
                       previous_history, current_history, minimum_history):
    if minimum_history < 1:
        raise ValueError("minimum_history must be positive.")
    scores = preference_shift_scores(before_user_embeddings, after_user_embeddings, centroids)
    previous = previous_history.user_items
    current = current_history.user_items
    existing = sorted(previous)
    if any(user < 0 or user >= len(scores) for user in existing):
        raise ValueError("History user IDs exceed the reference embeddings.")
    eligible = [user for user in existing if len(current.get(user, set())) >= minimum_history]
    values = scores.detach().cpu().numpy()
    ordered = sorted(eligible, key=lambda user: (float(values[user]), user))
    midpoint = len(ordered) // 2
    return {
        "stable": ordered[:midpoint],
        "dynamic": ordered[midpoint:],
        "drift": {user: float(values[user]) for user in eligible},
        "excluded_new_users": len(set(current).difference(previous)),
        "excluded_insufficient_history": len(existing) - len(eligible),
        "minimum_history": int(minimum_history),
    }


def deterministic_topk(scores, item_ids, k=10):
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(item_ids, torch.Tensor):
        item_ids = item_ids.detach().cpu().numpy()
    values = np.asarray(scores, dtype=np.float64)
    items = np.asarray(item_ids, dtype=np.int64)
    if values.ndim != 1 or items.ndim != 1 or values.shape != items.shape:
        raise ValueError("Scores and item IDs must be aligned vectors.")
    if k < 1 or len(items) < k:
        raise ValueError("A Top-k audit requires at least k eligible items.")
    if len(np.unique(items)) != len(items) or np.any(items < 0):
        raise ValueError("Audit item IDs must be distinct and nonnegative.")
    if not np.isfinite(values).all():
        raise FloatingPointError("Audit scores must be finite.")
    order = np.lexsort((items, -values))[:k]
    return items[order]


def entering_competitor_coverage(before_scores, after_scores, pool_items, candidate_items, anchor, k=10):
    pool = set(map(int, pool_items))
    candidates = set(map(int, candidate_items))
    if int(anchor) not in pool or int(anchor) not in candidates or not candidates.issubset(pool):
        raise ValueError("Candidate supports must contain the anchor and lie within the proposal pool.")
    before = set(map(int, deterministic_topk(before_scores, pool_items, k)))
    after = set(map(int, deterministic_topk(after_scores, pool_items, k)))
    entrants = after.difference(before).difference({int(anchor)})
    covered = entrants.intersection(candidates)
    return {
        "coverage": len(covered) / len(entrants) if entrants else None,
        "entering_count": len(entrants),
        "covered_count": len(covered),
        "entering_items": sorted(entrants),
    }


def ranking_turnover(before_scores, after_scores, item_ids, k=10):
    before = set(map(int, deterministic_topk(before_scores, item_ids, k)))
    after = set(map(int, deterministic_topk(after_scores, item_ids, k)))
    return 1.0 - len(before.intersection(after)) / k


def full_catalogue_eligible_items(num_items, history, user, anchor=None):
    if num_items < 1:
        raise ValueError("The catalogue must contain at least one item.")
    mask = np.ones(int(num_items), dtype=bool)
    seen = sorted(history.user_items.get(int(user), set()))
    if seen and (seen[0] < 0 or seen[-1] >= num_items):
        raise ValueError("History item IDs exceed the catalogue.")
    mask[seen] = False
    if anchor is not None:
        if not 0 <= int(anchor) < num_items:
            raise ValueError("The anchor lies outside the catalogue.")
        mask[int(anchor)] = True
    return np.flatnonzero(mask)


def _audit_embeddings(before_model, after_model, graph, masks):
    modes = before_model.training, after_model.training
    before_model.eval()
    after_model.eval()
    try:
        before = before_model.encode(graph, masks=masks)
        after = after_model.encode(graph, masks=masks)
    finally:
        before_model.train(modes[0])
        after_model.train(modes[1])
    if before[0].shape != after[0].shape or before[1].shape != after[1].shape:
        raise ValueError("Audit models must have matching user and item representations.")
    return before, after


@torch.no_grad()
def evaluate_competitor_coverage(before_model, after_model, graph, replay_edges,
                                 proposal_pools, candidate_sets, k=10, masks=None):
    if len(replay_edges) != len(proposal_pools) or len(replay_edges) != len(candidate_sets):
        raise ValueError("Every replay interaction requires a proposal pool and candidate support.")
    before, after = _audit_embeddings(before_model, after_model, graph, masks)
    records, insufficient, empty = [], 0, 0
    for edge, pool, candidates in zip(replay_edges, proposal_pools, candidate_sets):
        user, anchor = map(int, edge[:2])
        pool = np.asarray(pool, dtype=np.int64)
        if len(pool) < k:
            insufficient += 1
            continue
        ids = torch.as_tensor(pool, dtype=torch.long, device=before[1].device)
        pre_scores = before[0][user] @ before[1][ids].T
        post_scores = after[0][user] @ after[1][ids].T
        record = entering_competitor_coverage(pre_scores, post_scores, pool, candidates, anchor, k)
        if record["coverage"] is None:
            empty += 1
            continue
        records.append({"user": user, "anchor": anchor, **record})
    return {
        "mean_coverage": float(np.mean([record["coverage"] for record in records])) if records else None,
        "evaluated_interactions": len(records),
        "excluded_empty_entrants": empty,
        "excluded_insufficient_items": insufficient,
        "per_interaction": records,
    }


@torch.no_grad()
def evaluate_replay_turnover(before_model, after_model, graph, replay_edges, history, k=10, masks=None):
    before, after = _audit_embeddings(before_model, after_model, graph, masks)
    records, insufficient = [], 0
    for user, anchor, _ in replay_edges:
        user, anchor = int(user), int(anchor)
        eligible = full_catalogue_eligible_items(before_model.n_items, history, user, anchor)
        if len(eligible) < k:
            insufficient += 1
            continue
        ids = torch.as_tensor(eligible, dtype=torch.long, device=before[1].device)
        pre_scores = before[0][user] @ before[1][ids].T
        post_scores = after[0][user] @ after[1][ids].T
        turnover = ranking_turnover(pre_scores, post_scores, eligible, k)
        records.append({"user": user, "anchor": anchor, "turnover": turnover,
                        "eligible_items": len(eligible)})
    return {
        "mean_turnover": float(np.mean([record["turnover"] for record in records])) if records else None,
        "evaluated_interactions": len(records),
        "excluded_insufficient_items": insufficient,
        "per_interaction": records,
    }
