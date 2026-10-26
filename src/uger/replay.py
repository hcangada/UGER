from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import merge_edges, sample_recommendation_batch


@dataclass
class ReplaySelection:
    edges: np.ndarray
    model_scores: np.ndarray
    structural_scores: np.ndarray
    fusion_scores: np.ndarray
    capacity: int


def minmax_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values.copy()
    if not np.isfinite(values).all():
        raise FloatingPointError("Replay importance scores must be finite")
    lower, upper = values.min(), values.max()
    return (values - lower) / (upper - lower) if upper > lower else np.zeros_like(values)


def structural_scores(pool):
    if not len(pool):
        return np.empty(0, dtype=np.float64)
    pairs = np.unique(pool[:, :2], axis=0)
    users, user_counts = np.unique(pairs[:, 0], return_counts=True)
    items, item_counts = np.unique(pairs[:, 1], return_counts=True)
    user_degree = dict(zip(users.tolist(), user_counts.tolist()))
    item_degree = dict(zip(items.tolist(), item_counts.tolist()))
    return np.asarray([
        0.5 * (1.0 / user_degree[int(user)] + 1.0 / item_degree[int(item)])
        for user, item in pool[:, :2]
    ], dtype=np.float64)


def interaction_losses(model, batch, graph, masks=None):
    scores = model(batch["users"], batch["items"], graph, masks=masks)
    return F.softplus(scores[:, 1:] - scores[:, :1]).mean(dim=1)


def gradient_alignment_scores(model, pool, graph, history, num_negatives, batch_size, rng):
    device = next(model.parameters()).device
    sampled = sample_recommendation_batch(pool, history, num_negatives, rng, device=device)
    output = np.zeros(len(pool), dtype=np.float64)
    if not len(sampled["users"]):
        return output
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    aggregate = [torch.zeros_like(parameter) for parameter in parameters]
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(sampled["users"]), batch_size):
            batch = {key: value[start:start + batch_size] for key, value in sampled.items()}
            losses = interaction_losses(model, batch, graph)
            gradients = torch.autograd.grad(losses.sum(), parameters, allow_unused=True)
            for total, gradient in zip(aggregate, gradients):
                if gradient is not None:
                    total.add_(gradient.detach())
        for start in range(0, len(sampled["users"]), batch_size):
            batch = {key: value[start:start + batch_size] for key, value in sampled.items()}
            losses = interaction_losses(model, batch, graph)
            weights = torch.ones_like(losses, requires_grad=True)
            gradients = torch.autograd.grad(
                (losses * weights).sum(), parameters, create_graph=True, allow_unused=True
            )
            products = [
                (gradient * total).sum()
                for gradient, total in zip(gradients, aggregate)
                if gradient is not None
            ]
            if products:
                scores = torch.autograd.grad(sum(products), weights, allow_unused=True)[0]
                if scores is not None:
                    output[batch["row_indices"].cpu().numpy()] = scores.detach().double().cpu().numpy()
    finally:
        model.train(was_training)
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite model influence during replay selection")
    return output


def select_replay_buffer(model, pool, graph, history, capacity, beta,
                         num_negatives, batch_size, rng, method="UGER"):
    pool = merge_edges(np.asarray(pool, dtype=np.int64).reshape(-1, 3))
    if int(capacity) != capacity or capacity < 0:
        raise ValueError("Replay capacity must be a nonnegative integer")
    if not 0 <= beta <= 1:
        raise ValueError("The fusion weight must lie in [0, 1]")
    capacity = min(int(capacity), len(pool))
    model_scores = np.zeros(len(pool), dtype=np.float64)
    structure = structural_scores(pool)
    scores = np.zeros(len(pool), dtype=np.float64)
    if not capacity:
        return ReplaySelection(pool[:0].copy(), model_scores, structure, scores, capacity)
    if method == "Uniform":
        selected = rng.choice(len(pool), size=capacity, replace=False)
    else:
        if method == "INFER" or (method == "UGER" and beta > 0):
            model_scores = gradient_alignment_scores(
                model, pool, graph, history, num_negatives, batch_size, rng
            )
        if method == "Invdeg":
            probabilities = structure / structure.sum()
            selected = rng.choice(len(pool), size=capacity, replace=False, p=probabilities)
            scores = structure.copy()
        else:
            if method == "INFER":
                scores = minmax_normalize(model_scores)
            elif method == "UGER":
                scores = beta * minmax_normalize(model_scores) + (1.0 - beta) * minmax_normalize(structure)
            else:
                raise ValueError(f"Unsupported replay selection method: {method}")
            selected = np.lexsort((pool[:, 2], pool[:, 1], pool[:, 0], -scores))[:capacity]
    retained = pool[selected].copy()
    order = np.lexsort((retained[:, 1], retained[:, 0], retained[:, 2]))
    return ReplaySelection(retained[order], model_scores, structure, scores, capacity)
