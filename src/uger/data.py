from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def _edge_array(edges: np.ndarray | Iterable, columns: int = 3) -> np.ndarray:
    array = np.asarray(edges, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"Interactions must have shape (N, {columns}).")
    return array


def _timestamp(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        elapsed = instant.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return Decimal(elapsed.days * 86400 + elapsed.seconds) + Decimal(elapsed.microseconds) / Decimal(1_000_000)
    if not result.is_finite():
        raise ValueError("Timestamps must be finite.")
    magnitude = abs(result)
    if magnitude >= Decimal("1e17"):
        return result / Decimal(1_000_000_000)
    if magnitude >= Decimal("1e14"):
        return result / Decimal(1_000_000)
    if magnitude >= Decimal("1e11"):
        return result / Decimal(1000)
    return result


def read_interactions(path: str | Path, delimiter: str = "\t") -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    records = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=delimiter)
        if not {"user_id", "item_id", "timestamp"}.issubset(reader.fieldnames or []):
            raise ValueError("Input requires user_id, item_id, and timestamp columns.")
        for line, row in enumerate(reader, start=2):
            user, item = ((row.get(name, "") or "").strip() for name in ("user_id", "item_id"))
            if not user or not item:
                raise ValueError(f"Missing interaction ID on line {line}.")
            try:
                timestamp = _timestamp((row.get("timestamp", "") or "").strip())
            except (ValueError, InvalidOperation, OverflowError) as error:
                raise ValueError(f"Invalid timestamp on line {line}.") from error
            records.append((user, item, timestamp))
    if not records:
        raise ValueError("The interaction file is empty.")
    records.sort(key=lambda record: record[2])
    user_mapping, item_mapping = {}, {}
    edges = np.empty((len(records), 3), dtype=np.int64)
    for index, (user, item, timestamp) in enumerate(records):
        user_id = user_mapping.setdefault(user, len(user_mapping))
        item_id = item_mapping.setdefault(item, len(item_mapping))
        edges[index] = user_id, item_id, int((timestamp * Decimal(1_000_000)).to_integral_value(rounding=ROUND_FLOOR))
    return edges, user_mapping, item_mapping


@dataclass(frozen=True)
class InteractionStream:
    edges: np.ndarray
    snapshots: list[np.ndarray]
    initial_train: np.ndarray
    initial_validation: np.ndarray
    num_users: int
    num_items: int
    user_mapping: dict[str, int]
    item_mapping: dict[str, int]
    fingerprint: str

    @property
    def prediction_windows(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [split_prediction_window(edges) for edges in self.snapshots[1:]]


def split_prediction_window(edges: np.ndarray | Iterable) -> tuple[np.ndarray, np.ndarray]:
    array = _edge_array(edges)
    ordered = array[np.argsort(array[:, 2], kind="stable")]
    midpoint = len(ordered) // 2
    return ordered[:midpoint].copy(), ordered[midpoint:].copy()


def load_interaction_stream(
    path: str | Path,
    initial_fraction: float = 0.6,
    num_incremental: int = 5,
    validation_fraction: float = 0.1,
    split_mode: str = "count",
    delimiter: str = "\t",
) -> InteractionStream:
    if not 0 < initial_fraction < 1:
        raise ValueError("initial_fraction must lie strictly between zero and one.")
    if num_incremental < 1:
        raise ValueError("num_incremental must be positive.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must lie in [0, 1).")
    if split_mode not in {"count", "time"}:
        raise ValueError("split_mode must be 'count' or 'time'.")
    edges, users, items = read_interactions(path, delimiter)
    edges = merge_edges(edges)
    if len(edges) < 2:
        raise ValueError("At least two interactions are required.")
    initial_count = min(max(int(len(edges) * initial_fraction), 1), len(edges) - 1)
    initial, remainder = edges[:initial_count], edges[initial_count:]
    if split_mode == "count":
        increments = list(np.array_split(remainder, num_incremental))
    else:
        boundaries = np.linspace(int(remainder[0, 2]), int(remainder[-1, 2]) + 1, num_incremental + 1)
        indices = np.searchsorted(remainder[:, 2], boundaries[1:-1], side="left")
        increments = list(np.split(remainder, indices))
    validation_count = 0
    if validation_fraction > 0 and len(initial) > 1:
        validation_count = min(max(int(len(initial) * validation_fraction), 1), len(initial) - 1)
    cut = len(initial) - validation_count
    return InteractionStream(
        edges=edges,
        snapshots=[initial, *increments],
        initial_train=initial[:cut],
        initial_validation=initial[cut:],
        num_users=len(users),
        num_items=len(items),
        user_mapping=users,
        item_mapping=items,
        fingerprint=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


class ObservedHistory:
    def __init__(self, edges: np.ndarray | Iterable | None = None):
        self.user_items: dict[int, set[int]] = {}
        self._items: set[int] = set()
        self._eligible_items = np.empty(0, dtype=np.int64)
        if edges is not None:
            self.add(edges)

    @property
    def eligible_items(self) -> np.ndarray:
        return self._eligible_items.copy()

    def add(self, edges: np.ndarray | Iterable) -> None:
        array = np.asarray(edges, dtype=np.int64)
        if array.size == 0:
            return
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("Interactions must provide user and item IDs.")
        if np.any(array[:, :2] < 0):
            raise ValueError("Interaction IDs must be nonnegative.")
        for user, item in array[:, :2]:
            self.user_items.setdefault(int(user), set()).add(int(item))
            self._items.add(int(item))
        self._eligible_items = np.asarray(sorted(self._items), dtype=np.int64)

    def available_items(self, user: int, exclude: Iterable[int] = ()) -> np.ndarray:
        blocked = self.user_items.get(int(user), set()).union(int(item) for item in exclude)
        return np.asarray([item for item in self._eligible_items if int(item) not in blocked], dtype=np.int64)

    def unobserved_count(self, user: int, exclude: Iterable[int] = ()) -> int:
        blocked = self.user_items.get(int(user), set()).union(int(item) for item in exclude)
        return len(self._items) - len(self._items.intersection(blocked))

    def sample_unobserved(
        self,
        user: int,
        size: int,
        rng: np.random.Generator,
        replace: bool = False,
        exclude: Iterable[int] = (),
    ) -> np.ndarray:
        if size < 0:
            raise ValueError("Sample size must be nonnegative.")
        blocked = self.user_items.get(int(user), set()).union(int(item) for item in exclude)
        available_count = len(self._items) - len(self._items.intersection(blocked))
        if size == 0 or available_count == 0:
            return np.empty(0, dtype=np.int64)
        count = size if replace else min(size, available_count)
        chosen = []
        selected = set()
        budget = max(128, count * 16)
        while len(chosen) < count and budget > 0:
            draw_count = min(max(16, (count - len(chosen)) * 2), budget)
            proposals = rng.choice(self._eligible_items, size=draw_count, replace=True)
            budget -= draw_count
            for proposal in proposals:
                item = int(proposal)
                if item in blocked or (not replace and item in selected):
                    continue
                chosen.append(item)
                selected.add(item)
                if len(chosen) == count:
                    break
        if len(chosen) < count:
            remaining = np.asarray([item for item in self._eligible_items if int(item) not in blocked and (replace or int(item) not in selected)], dtype=np.int64)
            chosen.extend(rng.choice(remaining, size=count - len(chosen), replace=replace).tolist())
        return np.asarray(chosen, dtype=np.int64)


def sample_recommendation_batch(
    edges: np.ndarray | Iterable,
    history: ObservedHistory,
    num_negatives: int,
    rng: np.random.Generator,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    if num_negatives < 0:
        raise ValueError("num_negatives must be nonnegative.")
    array = _edge_array(edges)
    users, items, timestamps, row_indices = [], [], [], []
    for index, (user, positive, timestamp) in enumerate(array):
        available_count = history.unobserved_count(int(user), exclude=(int(positive),))
        if num_negatives and available_count == 0:
            continue
        negatives = history.sample_unobserved(int(user), num_negatives, rng, replace=available_count < num_negatives, exclude=(int(positive),))
        users.append(int(user))
        items.append([int(positive), *map(int, negatives)])
        timestamps.append(int(timestamp))
        row_indices.append(index)
    return {
        "users": torch.tensor(users, dtype=torch.long, device=device),
        "items": torch.tensor(items, dtype=torch.long, device=device).reshape(-1, 1 + num_negatives),
        "timestamps": torch.tensor(timestamps, dtype=torch.long, device=device),
        "row_indices": torch.tensor(row_indices, dtype=torch.long, device=device),
    }


def merge_edges(*arrays: np.ndarray | Iterable) -> np.ndarray:
    valid = [_edge_array(array) for array in arrays]
    if not valid or not any(len(array) for array in valid):
        return np.empty((0, 3), dtype=np.int64)
    merged = np.concatenate(valid)
    _, first_indices = np.unique(merged, axis=0, return_index=True)
    merged = merged[np.sort(first_indices)]
    return merged[np.argsort(merged[:, 2], kind="stable")]


def build_normalized_graph(
    edges: np.ndarray | Iterable,
    num_users: int,
    num_items: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if num_users < 0 or num_items < 0:
        raise ValueError("Graph dimensions must be nonnegative.")
    array = np.asarray(edges, dtype=np.int64)
    size = num_users + num_items
    if array.size == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty(0, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (size, size), device=device).coalesce()
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("Interactions must provide user and item IDs.")
    pairs = np.unique(array[:, :2], axis=0)
    if np.any(pairs < 0) or np.any(pairs[:, 0] >= num_users) or np.any(pairs[:, 1] >= num_items):
        raise ValueError("Interaction IDs exceed graph dimensions.")
    user_nodes, item_nodes = pairs[:, 0], num_users + pairs[:, 1]
    rows = np.concatenate((user_nodes, item_nodes))
    columns = np.concatenate((item_nodes, user_nodes))
    degrees = np.bincount(rows, minlength=size)
    values = 1.0 / np.sqrt(degrees[rows] * degrees[columns])
    indices = torch.as_tensor(np.stack((rows, columns)), dtype=torch.long, device=device)
    weights = torch.as_tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, weights, (size, size), device=device).coalesce()
