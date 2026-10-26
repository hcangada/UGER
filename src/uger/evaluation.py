from collections import defaultdict

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, graph, target_edges, history, top_k=(10, 20), batch_size=128,
             users=None, return_per_user=False):
    top_k = tuple(sorted({int(k) for k in top_k}))
    if not top_k or min(top_k) < 1:
        raise ValueError("top_k must contain positive integers.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    cohort = None if users is None else set(map(int, users))
    targets = defaultdict(set)
    catalogue_size = int(model.n_items)
    known_items = set(map(int, history.eligible_items))
    cold_events = 0
    repeated_events = 0
    cohort_events = 0
    for user, item, _ in target_edges:
        user, item = int(user), int(item)
        if not 0 <= user < model.n_users or not 0 <= item < catalogue_size:
            raise ValueError("Evaluation interaction IDs exceed model dimensions.")
        if cohort is not None and user not in cohort:
            cohort_events += 1
            continue
        if item in history.user_items.get(user, set()):
            repeated_events += 1
        else:
            targets[user].add(item)
            cold_events += int(user not in history.user_items or item not in known_items)
    device = next(model.parameters()).device
    catalog = torch.arange(catalogue_size, device=device, dtype=torch.long)
    metrics = {f"{metric}@{k}": 0.0 for metric in ("Recall", "NDCG") for k in top_k}
    evaluated_users = sorted(targets)
    per_user = {}
    was_training = model.training
    model.eval()
    try:
        user_embeddings, item_embeddings = model.encode(graph)
        catalog_embeddings = item_embeddings[catalog]
        for start in range(0, len(evaluated_users), batch_size):
            user_ids = evaluated_users[start:start + batch_size]
            ids = torch.tensor(user_ids, dtype=torch.long, device=device)
            scores = user_embeddings[ids] @ catalog_embeddings.T
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Evaluation scores must be finite before history filtering.")
            for row, user in enumerate(user_ids):
                seen = history.user_items.get(user, set())
                blocked = torch.as_tensor(sorted(seen), device=device, dtype=torch.long)
                if blocked.numel() and (blocked.min() < 0 or blocked.max() >= catalogue_size):
                    raise ValueError("History item IDs exceed the evaluation catalogue.")
                scores[row, blocked] = -torch.inf
                count = catalogue_size - len(seen)
                if not count:
                    continue
                ranking = torch.argsort(scores[row], descending=True, stable=True)[:min(max(top_k), count)]
                ranked = catalog[ranking].cpu().tolist()
                hits = np.asarray([item in targets[user] for item in ranked], dtype=np.float64)
                values = {}
                for k in top_k:
                    observed_hits = hits[:k]
                    values[f"Recall@{k}"] = float(observed_hits.sum() / len(targets[user]))
                    discounts = 1.0 / np.log2(np.arange(len(observed_hits)) + 2)
                    ideal = (1.0 / np.log2(np.arange(min(k, len(targets[user]))) + 2)).sum()
                    values[f"NDCG@{k}"] = float((observed_hits * discounts).sum() / ideal)
                for key, value in values.items():
                    metrics[key] += value
                if return_per_user:
                    per_user[user] = values
    finally:
        model.train(was_training)
    if evaluated_users:
        metrics = {key: value / len(evaluated_users) for key, value in metrics.items()}
    result = {"metrics": metrics, "evaluated_users": len(evaluated_users), "target_events": len(target_edges),
              "cold_events_excluded": 0, "cold_events_included": cold_events,
              "repeated_events_excluded": repeated_events, "cohort_events_excluded": cohort_events,
              "eligible_items": catalogue_size}
    if return_per_user:
        result["per_user"] = per_user
    return result
