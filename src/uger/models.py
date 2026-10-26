from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy import sparse
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _uniform(shape, device, generator=None):
    sampling_device = device if generator is None else generator.device
    return torch.rand(shape, device=sampling_device, generator=generator).to(device)


class BaseGraphRecommender(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 2,
        keep_prob: float = 1.0,
        message_dropout: float = 0.0,
    ):
        super().__init__()
        if min(n_users, n_items, embedding_dim) <= 0:
            raise ValueError("User, item, and embedding counts must be positive.")
        if n_layers < 0:
            raise ValueError("n_layers must be nonnegative.")
        if keep_prob == -1:
            keep_prob = 1.0
        if not 0.0 < keep_prob <= 1.0:
            raise ValueError("keep_prob must be in (0, 1], or -1 to disable dropout.")
        if not 0.0 <= message_dropout < 1.0:
            raise ValueError("message_dropout must be in [0, 1).")
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.user_num = self.n_users
        self.item_num = self.n_items
        self.embedding_dim = int(embedding_dim)
        self.emb_size = self.embedding_dim
        self.n_layers = int(n_layers)
        self.keep_prob = float(keep_prob)
        self.message_dropout = float(message_dropout)
        self.u_embeddings = nn.Embedding(self.n_users, self.embedding_dim)
        self.i_embeddings = nn.Embedding(self.n_items, self.embedding_dim)
        self.message_dimensions = (self.embedding_dim,) * self.n_layers
        nn.init.xavier_uniform_(self.u_embeddings.weight)
        nn.init.xavier_uniform_(self.i_embeddings.weight)

    def _coalesced_graph(self, graph: Tensor) -> Tensor:
        if not isinstance(graph, Tensor) or graph.layout == torch.strided:
            raise TypeError("graph must be a sparse PyTorch tensor.")
        size = self.n_users + self.n_items
        if graph.ndim != 2 or tuple(graph.shape) != (size, size):
            raise ValueError("graph shape must match the user and item counts.")
        reference = self.u_embeddings.weight
        if graph.device != reference.device:
            raise ValueError("graph and model parameters must be on the same device.")
        graph = graph.to(dtype=reference.dtype)
        if graph.layout != torch.sparse_coo:
            graph = graph.to_sparse_coo()
        return graph.coalesce()

    def sample_masks(self, graph: Tensor, generator=None) -> dict:
        graph = self._coalesced_graph(graph)
        if not self.training:
            return {}
        masks = {}
        if self.keep_prob < 1.0:
            masks["edge"] = _uniform(
                (graph._nnz(),), graph.device, generator
            ) < self.keep_prob
        if self.message_dropout > 0.0:
            keep = 1.0 - self.message_dropout
            masks["message"] = tuple(
                (
                    _uniform((graph.shape[0], width), graph.device, generator)
                    < keep
                ).to(dtype=self.u_embeddings.weight.dtype)
                / keep
                for width in self.message_dimensions
            )
        return masks

    def _masked_graph(self, graph: Tensor, masks: Mapping | None) -> tuple:
        graph = self._coalesced_graph(graph)
        masks = self.sample_masks(graph) if masks is None else masks
        edge_mask = masks.get("edge")
        if edge_mask is not None:
            if edge_mask.ndim != 1 or edge_mask.numel() != graph._nnz():
                raise ValueError("edge mask must match the coalesced graph entries.")
            if edge_mask.device != graph.device:
                raise ValueError("edge mask and graph must be on the same device.")
            if edge_mask.dtype != torch.bool:
                raise TypeError("edge mask must have boolean dtype.")
            graph = torch.sparse_coo_tensor(
                graph.indices()[:, edge_mask],
                graph.values()[edge_mask] / self.keep_prob,
                graph.shape,
                device=graph.device,
                dtype=graph.dtype,
            ).coalesce()
        return graph, masks

    def encode(self, graph: Tensor, masks: Mapping | None = None) -> tuple:
        raise NotImplementedError

    def forward(
        self,
        users: Tensor,
        items: Tensor,
        graph: Tensor,
        masks: Mapping | None = None,
    ) -> Tensor:
        if users.dtype != torch.long or items.dtype != torch.long:
            raise TypeError("users and items must have torch.long dtype.")
        if users.ndim != 1 or items.ndim != 2 or items.shape[0] != users.shape[0]:
            raise ValueError("users must have shape [B] and items shape [B, C].")
        user_embeddings, item_embeddings = self.encode(graph, masks)
        return (
            F.embedding(users, user_embeddings).unsqueeze(1)
            * F.embedding(items, item_embeddings)
        ).sum(dim=-1)

    def per_interaction_loss(
        self, batch: Mapping, graph: Tensor, masks: Mapping | None = None
    ) -> Tensor:
        users = batch.get("users", batch.get("user_id"))
        items = batch.get("items", batch.get("item_id"))
        if users is None or items is None:
            raise KeyError("batch must contain users and items.")
        if users.ndim == 2 and users.shape[1] == 1:
            users = users[:, 0]
        if items.ndim != 2 or items.shape[1] < 2:
            raise ValueError("BPR requires a positive item followed by negatives.")
        scores = self(users, items, graph, masks)
        return F.softplus(scores[:, 1:] - scores[:, :1]).mean(dim=1)

    def recommendation_loss(
        self, batch: Mapping, graph: Tensor, masks: Mapping | None = None
    ) -> Tensor:
        losses = self.per_interaction_loss(batch, graph, masks)
        if losses.numel() == 0:
            raise ValueError("recommendation_loss requires a nonempty batch.")
        return losses.mean()


class LightGCN(BaseGraphRecommender):
    def __init__(self, *args, message_dropout=0.0, **kwargs):
        if message_dropout != 0.0:
            raise ValueError("LightGCN uses graph dropout through keep_prob.")
        super().__init__(*args, message_dropout=0.0, **kwargs)

    def encode(self, graph: Tensor, masks: Mapping | None = None) -> tuple:
        graph, masks = self._masked_graph(graph, masks)
        embeddings = torch.cat(
            (self.u_embeddings.weight, self.i_embeddings.weight), dim=0
        )
        layers = [embeddings]
        for _ in range(self.n_layers):
            embeddings = torch.sparse.mm(graph, embeddings)
            layers.append(embeddings)
        embeddings = torch.stack(layers, dim=0).mean(dim=0)
        return embeddings[: self.n_users], embeddings[self.n_users :]


class NGCF(BaseGraphRecommender):
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        n_layers=2,
        keep_prob=1.0,
        message_dropout=0.0,
        layer_sizes: Sequence[int] | None = None,
        negative_slope=0.2,
        normalize_output=True,
    ):
        if layer_sizes is not None:
            layer_sizes = tuple(int(width) for width in layer_sizes)
            if len(layer_sizes) != n_layers or any(width <= 0 for width in layer_sizes):
                raise ValueError("layer_sizes must contain n_layers positive widths.")
        else:
            layer_sizes = (embedding_dim,) * n_layers
        super().__init__(
            n_users, n_items, embedding_dim, n_layers, keep_prob, message_dropout
        )
        self.message_dimensions = layer_sizes
        self.negative_slope = float(negative_slope)
        self.normalize_output = bool(normalize_output)
        widths = (embedding_dim,) + layer_sizes
        self.sum_layers = nn.ModuleList(
            nn.Linear(widths[layer], widths[layer + 1], bias=False)
            for layer in range(n_layers)
        )
        self.interaction_layers = nn.ModuleList(
            nn.Linear(widths[layer], widths[layer + 1], bias=False)
            for layer in range(n_layers)
        )
        for layer in (*self.sum_layers, *self.interaction_layers):
            nn.init.xavier_uniform_(layer.weight)

    def encode(self, graph: Tensor, masks: Mapping | None = None) -> tuple:
        graph, masks = self._masked_graph(graph, masks)
        embeddings = torch.cat(
            (self.u_embeddings.weight, self.i_embeddings.weight), dim=0
        )
        layers = [embeddings]
        message_masks = masks.get("message", ())
        if message_masks and len(message_masks) != self.n_layers:
            raise ValueError("message masks must contain one mask per layer.")
        for index, (sum_layer, interaction_layer) in enumerate(
            zip(self.sum_layers, self.interaction_layers)
        ):
            neighbors = torch.sparse.mm(graph, embeddings)
            embeddings = F.leaky_relu(
                sum_layer(embeddings + neighbors)
                + interaction_layer(embeddings * neighbors),
                negative_slope=self.negative_slope,
            )
            if message_masks:
                if message_masks[index].shape != embeddings.shape:
                    raise ValueError("message mask shape does not match layer output.")
                embeddings = embeddings * message_masks[index]
            layers.append(
                F.normalize(embeddings, p=2, dim=1)
                if self.normalize_output
                else embeddings
            )
        embeddings = torch.cat(layers, dim=1)
        return embeddings[: self.n_users], embeddings[self.n_users :]


class MGCCF(BaseGraphRecommender):
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        n_layers=2,
        keep_prob=1.0,
        message_dropout=0.0,
        neighbor_samples: Sequence[int] | None = None,
        similarity_degree=10,
        similarity_block_size=1024,
    ):
        super().__init__(
            n_users, n_items, embedding_dim, n_layers, keep_prob, message_dropout
        )
        if neighbor_samples is None:
            neighbor_samples = tuple(15 if index == 0 else 10 for index in range(n_layers))
        else:
            neighbor_samples = tuple(int(value) for value in neighbor_samples)
        if len(neighbor_samples) != n_layers or any(value <= 0 for value in neighbor_samples):
            raise ValueError("neighbor_samples must contain n_layers positive integers.")
        if not np.isfinite(similarity_degree) or similarity_degree < 0:
            raise ValueError("similarity_degree must be finite and nonnegative.")
        if similarity_block_size < 1:
            raise ValueError("similarity_block_size must be positive.")
        self.neighbor_samples = neighbor_samples
        self.similarity_degree = float(similarity_degree)
        self.similarity_block_size = int(similarity_block_size)
        self.user_aggregators = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(n_layers)
        )
        self.item_aggregators = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(n_layers)
        )
        self.user_layers = nn.ModuleList(
            nn.Linear(2 * embedding_dim, embedding_dim, bias=False) for _ in range(n_layers)
        )
        self.item_layers = nn.ModuleList(
            nn.Linear(2 * embedding_dim, embedding_dim, bias=False) for _ in range(n_layers)
        )
        self.user_similarity = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.item_similarity = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.user_skip = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.item_skip = nn.Linear(embedding_dim, embedding_dim, bias=False)
        for layers in (
            self.user_aggregators,
            self.item_aggregators,
            self.user_layers,
            self.item_layers,
            (self.user_similarity, self.item_similarity, self.user_skip, self.item_skip),
        ):
            for layer in layers:
                nn.init.xavier_uniform_(layer.weight)
        self._graph_cache_key = None
        self._graph_cache_source = None
        self._graph_cache = None

    def _similarity_graph(self, profiles, device, dtype):
        count = profiles.shape[0]
        budget = min(int(count * self.similarity_degree / 2), count * (count - 1) // 2)
        if budget == 0:
            indices = torch.empty((2, 0), dtype=torch.long, device=device)
            return torch.sparse_coo_tensor(
                indices, torch.empty(0, device=device, dtype=dtype),
                (count, count), device=device, dtype=dtype,
            ).coalesce()
        degrees = np.asarray(profiles.multiply(profiles).sum(axis=1)).ravel()
        inverse = np.zeros_like(degrees, dtype=np.float64)
        np.divide(1.0, np.sqrt(degrees), out=inverse, where=degrees > 0)
        normalized = sparse.diags(inverse) @ profiles
        normalized = normalized.tocsr()
        retained_rows = np.empty(0, dtype=np.int64)
        retained_columns = np.empty(0, dtype=np.int64)
        retained_scores = np.empty(0, dtype=np.float64)
        for start in range(0, count, self.similarity_block_size):
            block = (normalized[start:start + self.similarity_block_size] @ normalized.T).tocoo()
            rows = block.row.astype(np.int64) + start
            columns = block.col.astype(np.int64)
            keep = (rows < columns) & (block.data > 0)
            rows, columns, scores = rows[keep], columns[keep], block.data[keep]
            if rows.size == 0:
                continue
            rows = np.concatenate((retained_rows, rows))
            columns = np.concatenate((retained_columns, columns))
            scores = np.concatenate((retained_scores, scores))
            order = np.lexsort((columns, rows, -scores))[:budget]
            retained_rows, retained_columns, retained_scores = rows[order], columns[order], scores[order]
        rows = np.concatenate((retained_rows, retained_columns))
        columns = np.concatenate((retained_columns, retained_rows))
        indices = torch.as_tensor(np.stack((rows, columns)), dtype=torch.long, device=device)
        values = torch.ones(rows.size, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(indices, values, (count, count), device=device).coalesce()

    def _context(self, graph):
        graph = self._coalesced_graph(graph)
        key = (id(graph), graph._version, graph.device, graph.dtype, graph._nnz())
        if self._graph_cache_key == key:
            return self._graph_cache
        indices = graph.indices()
        crosses = (indices[0] < self.n_users) & (indices[1] >= self.n_users)
        rows = indices[0, crosses].detach().cpu().numpy()
        columns = indices[1, crosses].detach().cpu().numpy() - self.n_users
        interactions = sparse.csr_matrix(
            (np.ones(rows.size), (rows, columns)), shape=(self.n_users, self.n_items)
        )
        interactions.data.fill(1.0)
        context = {
            "user_similarity": self._similarity_graph(interactions, graph.device, graph.dtype),
            "item_similarity": self._similarity_graph(interactions.T.tocsr(), graph.device, graph.dtype),
        }
        self._graph_cache_source = graph
        self._graph_cache_key = key
        self._graph_cache = context
        return context

    def _neighbor_mask(self, graph, quota, generator):
        rows = graph.indices()[0]
        degrees = torch.bincount(rows, minlength=graph.shape[0])
        selected = degrees[rows] <= quota
        remaining = torch.nonzero(~selected, as_tuple=False).flatten()
        if remaining.numel() == 0:
            return selected
        random_order = torch.argsort(_uniform((remaining.numel(),), graph.device, generator), stable=True)
        remaining = remaining[random_order]
        remaining = remaining[torch.argsort(rows[remaining], stable=True)]
        positions = torch.arange(remaining.numel(), device=graph.device)
        starts = torch.ones_like(positions, dtype=torch.bool)
        starts[1:] = rows[remaining[1:]] != rows[remaining[:-1]]
        offsets = torch.where(starts, positions, torch.zeros_like(positions)).cummax(dim=0).values
        selected[remaining[positions - offsets < quota]] = True
        return selected

    def sample_masks(self, graph: Tensor, generator=None) -> dict:
        graph = self._coalesced_graph(graph)
        masks = super().sample_masks(graph, generator)
        if self.training:
            masks["neighbors"] = tuple(
                self._neighbor_mask(graph, quota, generator) for quota in self.neighbor_samples
            )
        return masks

    def _mean_graph(self, graph, masks, layer):
        selected = masks.get("edge")
        if selected is None:
            selected = torch.ones(graph._nnz(), dtype=torch.bool, device=graph.device)
        elif selected.dtype != torch.bool or selected.shape != (graph._nnz(),):
            raise ValueError("edge mask must match the coalesced graph entries.")
        elif selected.device != graph.device:
            raise ValueError("edge mask and graph must be on the same device.")
        neighbors = masks.get("neighbors", ())
        if neighbors:
            if len(neighbors) != self.n_layers:
                raise ValueError("neighbor masks must contain one mask per layer.")
            neighbor = neighbors[layer]
            if neighbor.dtype != torch.bool or neighbor.shape != selected.shape or neighbor.device != graph.device:
                raise ValueError("neighbor mask must match the coalesced graph entries.")
            selected = selected & neighbor
        indices = graph.indices()[:, selected]
        degrees = torch.bincount(indices[0], minlength=graph.shape[0]).clamp_min(1)
        values = degrees[indices[0]].to(graph.dtype).reciprocal()
        return torch.sparse_coo_tensor(indices, values, graph.shape, device=graph.device).coalesce()

    def encode(self, graph: Tensor, masks: Mapping | None = None) -> tuple:
        graph = self._coalesced_graph(graph)
        context = self._context(graph)
        masks = self.sample_masks(graph) if masks is None else masks
        message_masks = masks.get("message", ())
        if message_masks and len(message_masks) != self.n_layers:
            raise ValueError("message masks must contain one mask per layer.")
        users, items = self.u_embeddings.weight, self.i_embeddings.weight
        initial_users, initial_items = users, items
        for index in range(self.n_layers):
            mean_graph = self._mean_graph(graph, masks, index)
            neighbors = torch.sparse.mm(mean_graph, torch.cat((users, items), dim=0))
            user_neighbors = torch.tanh(self.user_aggregators[index](neighbors[:self.n_users]))
            item_neighbors = torch.tanh(self.item_aggregators[index](neighbors[self.n_users:]))
            users = torch.tanh(self.user_layers[index](torch.cat((users, user_neighbors), dim=1)))
            items = torch.tanh(self.item_layers[index](torch.cat((items, item_neighbors), dim=1)))
            if message_masks:
                if message_masks[index].shape != (self.n_users + self.n_items, self.embedding_dim):
                    raise ValueError("message mask shape does not match layer output.")
                users = users * message_masks[index][:self.n_users]
                items = items * message_masks[index][self.n_users:]
        user_proximity = torch.tanh(self.user_similarity(
            torch.sparse.mm(context["user_similarity"], initial_users)
        ))
        item_proximity = torch.tanh(self.item_similarity(
            torch.sparse.mm(context["item_similarity"], initial_items)
        ))
        return (
            users + user_proximity + self.user_skip(initial_users),
            items + item_proximity + self.item_skip(initial_items),
        )


class MatrixFactorization(BaseGraphRecommender):
    def __init__(self, n_users, n_items, embedding_dim=64, **kwargs):
        kwargs.pop("n_layers", None)
        kwargs.pop("message_dropout", None)
        super().__init__(
            n_users, n_items, embedding_dim, n_layers=0, message_dropout=0.0, **kwargs
        )

    def encode(self, graph: Tensor, masks: Mapping | None = None) -> tuple:
        return self.u_embeddings.weight, self.i_embeddings.weight

    def sample_masks(self, graph: Tensor, generator=None) -> dict:
        return {}


def _model_key(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


MODEL_REGISTRY: dict[str, Callable] = {}


def register_model(name: str, factory=None, *, replace=False):
    key = _model_key(name)

    def register(candidate):
        if key in MODEL_REGISTRY and not replace:
            raise ValueError(f"A model is already registered as {name}.")
        if not callable(candidate):
            raise TypeError("A model factory must be callable.")
        MODEL_REGISTRY[key] = candidate
        return candidate

    return register if factory is None else register(factory)


def build_model(
    name: str,
    n_users: int,
    n_items: int,
    embedding_dim: int = 64,
    n_layers: int = 2,
    keep_prob: float = 1.0,
    message_dropout: float = 0.0,
    **kwargs,
) -> BaseGraphRecommender:
    key = _model_key(name)
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name}; available models: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[key](
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        keep_prob=keep_prob,
        message_dropout=message_dropout,
        **kwargs,
    )
    if not isinstance(model, BaseGraphRecommender):
        raise TypeError("Registered adapters must extend BaseGraphRecommender.")
    return model


for _name, _factory in (
    ("lightgcn", LightGCN),
    ("lgn", LightGCN),
    ("ngcf", NGCF),
    ("mgccf", MGCCF),
    ("multigccf", MGCCF),
    ("mf", MatrixFactorization),
    ("matrixfactorization", MatrixFactorization),
):
    register_model(_name, _factory)
