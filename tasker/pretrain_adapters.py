from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from typing import Any, Callable, Optional

import torch
from torch_geometric.data import DataLoader

from .adapters import BaseTaskAdapter
from .base import ContrastiveQuery, PartialHyperedgeQuery, TaskBatch, TaskType


def _get_x(data: Any) -> torch.Tensor:
    if hasattr(data, "x"):
        return data.x
    if hasattr(data, "data") and hasattr(data.data, "x"):
        return data.data.x
    raise AttributeError("data must expose x or data.x")


def _get_num_nodes(data: Any) -> int:
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        return int(data.num_nodes)
    if hasattr(data, "num_nodes") and isinstance(data.num_nodes, torch.Tensor):
        return int(data.num_nodes.item())
    if hasattr(data, "n_x"):
        n_x = data.n_x
        return int(n_x.item()) if isinstance(n_x, torch.Tensor) else int(n_x)
    return int(_get_x(data).shape[0])


def _get_hyperedge_index(data: Any) -> torch.Tensor:
    for attr in ("hyperedge_index", "edge_index"):
        if hasattr(data, attr):
            return getattr(data, attr)
    if hasattr(data, "data"):
        return _get_hyperedge_index(data.data)
    raise AttributeError("data must expose hyperedge_index, edge_index, or data.hyperedge_index")


def _clone_graph(data: Any, hyperedge_index: torch.Tensor, x: Optional[torch.Tensor] = None) -> Any:
    if hasattr(data, "clone"):
        graph = data.clone()
    else:
        graph = copy.copy(data)
        if hasattr(data, "data"):
            graph.data = copy.copy(data.data)

    graph.hyperedge_index = hyperedge_index
    graph.edge_index = hyperedge_index
    if x is not None:
        graph.x = x
        if hasattr(graph, "data"):
            graph.data.x = x
    if hasattr(graph, "norm"):
        graph.norm = torch.ones_like(hyperedge_index[0], dtype=torch.float, device=hyperedge_index.device)
    if hasattr(graph, "data"):
        graph.data.hyperedge_index = hyperedge_index
        graph.data.edge_index = hyperedge_index
        if hasattr(graph.data, "norm"):
            graph.data.norm = torch.ones_like(hyperedge_index[0], dtype=torch.float, device=hyperedge_index.device)
    return graph


def _maybe_transform_graph(data: Any, transform: Optional[Callable[[Any], Any]]) -> Any:
    if transform is None:
        return data
    return transform(data)


def _hyperedges_from_index(
    hyperedge_index: torch.Tensor,
    min_size: int = 2,
) -> list[tuple[int, torch.Tensor]]:
    hyperedges: dict[int, list[int]] = {}
    for node_id, edge_id in zip(hyperedge_index[0].detach().cpu().tolist(), hyperedge_index[1].detach().cpu().tolist()):
        hyperedges.setdefault(int(edge_id), []).append(int(node_id))

    result = []
    for edge_id, nodes in hyperedges.items():
        uniq_nodes = sorted(set(nodes))
        if len(uniq_nodes) >= min_size:
            result.append((edge_id, torch.tensor(uniq_nodes, dtype=torch.long)))
    return result


def _mask_memberships(
    hyperedge_index: torch.Tensor,
    memberships: list[tuple[int, int]],
) -> torch.Tensor:
    if not memberships:
        return hyperedge_index

    keep = torch.ones(hyperedge_index.shape[1], dtype=torch.bool, device=hyperedge_index.device)
    for node_id, edge_id in memberships:
        keep &= ~((hyperedge_index[0] == node_id) & (hyperedge_index[1] == edge_id))
    return hyperedge_index[:, keep]


def _sample_negative_nodes(
    num_nodes: int,
    positive_edge_nodes: torch.Tensor,
    num_negatives: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if num_negatives <= 0:
        return torch.empty(0, dtype=torch.long)

    blocked = set(positive_edge_nodes.detach().cpu().tolist())
    candidates = torch.tensor([node_id for node_id in range(num_nodes) if node_id not in blocked], dtype=torch.long)
    if candidates.numel() == 0:
        candidates = torch.arange(num_nodes, dtype=torch.long)

    if candidates.numel() >= num_negatives:
        perm = torch.randperm(candidates.numel(), generator=generator)[:num_negatives]
        return candidates[perm]

    sampled = torch.randint(candidates.numel(), (num_negatives,), generator=generator)
    return candidates[sampled]


def _sample_negative_nodes_from_candidates(
    candidate_pool: torch.Tensor,
    positive_edge_nodes: torch.Tensor,
    num_negatives: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if num_negatives <= 0:
        return torch.empty(0, dtype=torch.long, device=candidate_pool.device)

    blocked = set(positive_edge_nodes.detach().cpu().tolist())
    candidates = candidate_pool[
        torch.tensor(
            [int(node_id) not in blocked for node_id in candidate_pool.detach().cpu().tolist()],
            dtype=torch.bool,
            device=candidate_pool.device,
        )
    ]
    if candidates.numel() == 0:
        candidates = candidate_pool

    if candidates.numel() >= num_negatives:
        perm = torch.randperm(candidates.numel(), generator=generator)[:num_negatives].to(candidates.device)
        return candidates[perm].long()

    sampled = torch.randint(candidates.numel(), (num_negatives,), generator=generator).to(candidates.device)
    return candidates[sampled].long()


def _sample_indices(
    population_size: int,
    sample_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if population_size <= 0 or sample_size <= 0:
        return torch.empty(0, dtype=torch.long)
    if sample_size <= population_size:
        if shuffle:
            return torch.randperm(population_size, generator=generator)[:sample_size]
        return torch.arange(sample_size, dtype=torch.long)
    return torch.randint(population_size, (sample_size,), generator=generator)


class HyperedgeFillTaskAdapter(BaseTaskAdapter):
    task_type = TaskType.SSL_HYPEREDGE_FILL

    def __init__(
        self,
        data: Any,
        batch_size: int = 256,
        num_negatives: int = 15,
        samples_per_epoch: Optional[int] = None,
        min_hyperedge_size: int = 2,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
        graph_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.samples_per_epoch = samples_per_epoch
        self.min_hyperedge_size = min_hyperedge_size
        self.shuffle = shuffle
        self.generator = generator
        self.graph_transform = graph_transform

        self.hyperedge_index = _get_hyperedge_index(data)
        self.hyperedges = _hyperedges_from_index(self.hyperedge_index, min_size=min_hyperedge_size)
        self.num_nodes = _get_num_nodes(data)

    def __len__(self) -> int:
        sample_count = self.samples_per_epoch or len(self.hyperedges)
        return math.ceil(sample_count / self.batch_size) if sample_count > 0 else 0

    def __iter__(self) -> Iterator[TaskBatch]:
        sample_count = self.samples_per_epoch or len(self.hyperedges)
        if sample_count <= 0:
            return

        selected = _sample_indices(len(self.hyperedges), sample_count, self.shuffle, self.generator)
        for start in range(0, selected.numel(), self.batch_size):
            batch_indices = selected[start : start + self.batch_size]

            contexts = []
            candidate_rows = []
            target_nodes = []
            hyperedge_ids = []
            masked_memberships = []

            for hyperedge_pos in batch_indices.tolist():
                edge_id, edge_nodes = self.hyperedges[hyperedge_pos]
                target_pos = torch.randint(edge_nodes.numel(), (1,), generator=self.generator).item()
                target_node = int(edge_nodes[target_pos].item())
                context = edge_nodes[torch.arange(edge_nodes.numel()) != target_pos]
                negatives = _sample_negative_nodes(self.num_nodes, edge_nodes, self.num_negatives, self.generator)
                candidates = torch.cat([torch.tensor([target_node], dtype=torch.long), negatives], dim=0)

                contexts.append(context)
                candidate_rows.append(candidates)
                target_nodes.append(target_node)
                hyperedge_ids.append(edge_id)
                masked_memberships.append((target_node, edge_id))

            h_prime_index = _mask_memberships(self.hyperedge_index, masked_memberships)
            h_prime = _maybe_transform_graph(_clone_graph(self.data, h_prime_index), self.graph_transform)
            query = PartialHyperedgeQuery(
                contexts=contexts,
                candidate_nodes=torch.stack(candidate_rows, dim=0),
                hyperedge_ids=torch.tensor(hyperedge_ids, dtype=torch.long),
                target_nodes=torch.tensor(target_nodes, dtype=torch.long),
            )
            labels = torch.zeros(len(contexts), dtype=torch.long)

            yield TaskBatch(
                h_prime=h_prime,
                query=query,
                task_type=self.task_type,
                y=labels,
                split="pretrain",
                metadata={
                    "target_nodes": query.target_nodes,
                    "masked_memberships": masked_memberships,
                    "label_semantics": "index of the positive candidate in each candidate row",
                },
            )


class HypergraphContrastTaskAdapter(BaseTaskAdapter):
    task_type = TaskType.SSL_CONTRAST

    def __init__(
        self,
        data: Any,
        batch_size: Optional[int] = 1024,
        views_per_epoch: int = 1,
        anchor_type: str = "node",
        drop_incidence_rate: float = 0.2,
        drop_feature_rate: float = 0.0,
        min_hyperedge_size: int = 2,
        generator: Optional[torch.Generator] = None,
        graph_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if anchor_type not in {"node", "hyperedge", "graph"}:
            raise ValueError("anchor_type must be 'node', 'hyperedge', or 'graph'")
        self.data = data
        self.batch_size = batch_size
        self.views_per_epoch = views_per_epoch
        self.anchor_type = anchor_type
        self.drop_incidence_rate = drop_incidence_rate
        self.drop_feature_rate = drop_feature_rate
        self.generator = generator
        self.graph_transform = graph_transform

        self.hyperedge_index = _get_hyperedge_index(data)
        self.num_nodes = _get_num_nodes(data)
        self.hyperedges = _hyperedges_from_index(self.hyperedge_index, min_size=min_hyperedge_size)

    def _anchors(self) -> Optional[torch.Tensor]:
        if self.anchor_type == "node":
            return torch.arange(self.num_nodes, dtype=torch.long)
        if self.anchor_type == "hyperedge":
            return torch.tensor([edge_id for edge_id, _ in self.hyperedges], dtype=torch.long)
        return None

    def __len__(self) -> int:
        if self.anchor_type == "graph":
            return self.views_per_epoch
        anchor_count = self._anchors().numel()
        batch_size = self.batch_size or anchor_count
        return self.views_per_epoch * math.ceil(anchor_count / batch_size) if anchor_count > 0 else 0

    def _augment_view(self) -> Any:
        hyperedge_index = self.hyperedge_index
        if self.drop_incidence_rate > 0:
            keep = torch.rand(hyperedge_index.shape[1], generator=self.generator)
            keep = (keep >= self.drop_incidence_rate).to(hyperedge_index.device)
            if keep.sum() == 0:
                keep[torch.randint(keep.numel(), (1,), generator=self.generator).item()] = True
            hyperedge_index = hyperedge_index[:, keep]

        x = None
        if self.drop_feature_rate > 0:
            x = _get_x(self.data).clone()
            mask = torch.rand(x.shape[0], generator=self.generator).to(x.device) < self.drop_feature_rate
            x[mask] = 0

        return _maybe_transform_graph(_clone_graph(self.data, hyperedge_index, x=x), self.graph_transform)

    def __iter__(self) -> Iterator[TaskBatch]:
        for view_id in range(self.views_per_epoch):
            view_a = self._augment_view()
            view_b = self._augment_view()

            if self.anchor_type == "graph":
                yield TaskBatch(
                    h_prime=(view_a, view_b),
                    query=ContrastiveQuery(anchor_type=self.anchor_type),
                    task_type=self.task_type,
                    y=None,
                    split="pretrain",
                    metadata={"view_id": view_id},
                )
                continue

            anchors = self._anchors()
            if anchors is None or anchors.numel() == 0:
                continue
            anchors = anchors[torch.randperm(anchors.numel(), generator=self.generator)]
            batch_size = self.batch_size or anchors.numel()
            for start in range(0, anchors.numel(), batch_size):
                anchor_ids = anchors[start : start + batch_size]
                yield TaskBatch(
                    h_prime=(view_a, view_b),
                    query=ContrastiveQuery(anchor_type=self.anchor_type, anchor_ids=anchor_ids),
                    task_type=self.task_type,
                    y=torch.arange(anchor_ids.numel(), dtype=torch.long),
                    split="pretrain",
                    metadata={
                        "view_id": view_id,
                        "label_semantics": "positive pairs lie on the diagonal after scoring view_a anchors against view_b anchors",
                    },
                )


class HypergraphDatasetFillTaskAdapter(BaseTaskAdapter):
    task_type = TaskType.SSL_HYPEREDGE_FILL

    def __init__(
        self,
        dataset: Any,
        graph_batch_size: int = 32,
        fill_batch_size: int = 256,
        num_negatives: int = 15,
        samples_per_graph_batch: Optional[int] = None,
        min_hyperedge_size: int = 2,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
        graph_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.dataset = dataset
        self.graph_batch_size = graph_batch_size
        self.fill_batch_size = fill_batch_size
        self.num_negatives = num_negatives
        self.samples_per_graph_batch = samples_per_graph_batch
        self.min_hyperedge_size = min_hyperedge_size
        self.shuffle = shuffle
        self.generator = generator
        self.graph_transform = graph_transform

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.graph_batch_size)

    def _node_pool(self, batch: Any, edge_nodes: torch.Tensor) -> torch.Tensor:
        if not hasattr(batch, "batch") or batch.batch is None:
            return torch.arange(_get_num_nodes(batch), dtype=torch.long, device=edge_nodes.device)
        graph_id = batch.batch[edge_nodes[0].to(batch.batch.device)]
        return torch.where(batch.batch == graph_id)[0].to(edge_nodes.device)

    def __iter__(self) -> Iterator[TaskBatch]:
        loader = DataLoader(self.dataset, batch_size=self.graph_batch_size, shuffle=self.shuffle)
        for graph_batch in loader:
            hyperedge_index = _get_hyperedge_index(graph_batch)
            hyperedges = _hyperedges_from_index(hyperedge_index, min_size=self.min_hyperedge_size)
            sample_count = self.samples_per_graph_batch or len(hyperedges)
            selected = _sample_indices(len(hyperedges), sample_count, self.shuffle, self.generator)

            for start in range(0, selected.numel(), self.fill_batch_size):
                batch_indices = selected[start : start + self.fill_batch_size]
                contexts = []
                candidate_rows = []
                target_nodes = []
                hyperedge_ids = []
                masked_memberships = []

                for hyperedge_pos in batch_indices.tolist():
                    edge_id, edge_nodes = hyperedges[hyperedge_pos]
                    edge_nodes = edge_nodes.to(hyperedge_index.device)
                    target_pos = torch.randint(edge_nodes.numel(), (1,), generator=self.generator).item()
                    target_node = int(edge_nodes[target_pos].item())
                    context = edge_nodes[torch.arange(edge_nodes.numel(), device=edge_nodes.device) != target_pos]
                    node_pool = self._node_pool(graph_batch, edge_nodes)
                    negatives = _sample_negative_nodes_from_candidates(
                        node_pool,
                        edge_nodes,
                        self.num_negatives,
                        self.generator,
                    ).cpu()
                    candidates = torch.cat([torch.tensor([target_node], dtype=torch.long), negatives], dim=0)

                    contexts.append(context.cpu())
                    candidate_rows.append(candidates)
                    target_nodes.append(target_node)
                    hyperedge_ids.append(edge_id)
                    masked_memberships.append((target_node, edge_id))

                if not contexts:
                    continue

                h_prime_index = _mask_memberships(hyperedge_index, masked_memberships)
                h_prime = _maybe_transform_graph(_clone_graph(graph_batch, h_prime_index), self.graph_transform)
                query = PartialHyperedgeQuery(
                    contexts=contexts,
                    candidate_nodes=torch.stack(candidate_rows, dim=0),
                    hyperedge_ids=torch.tensor(hyperedge_ids, dtype=torch.long),
                    target_nodes=torch.tensor(target_nodes, dtype=torch.long),
                )
                yield TaskBatch(
                    h_prime=h_prime,
                    query=query,
                    task_type=self.task_type,
                    y=torch.zeros(len(contexts), dtype=torch.long),
                    split="pretrain",
                    metadata={
                        "target_nodes": query.target_nodes,
                        "masked_memberships": masked_memberships,
                        "label_semantics": "index of the positive candidate in each candidate row",
                    },
                )


class HypergraphDatasetContrastTaskAdapter(BaseTaskAdapter):
    task_type = TaskType.SSL_CONTRAST

    def __init__(
        self,
        dataset: Any,
        graph_batch_size: int = 32,
        views_per_epoch: int = 1,
        drop_incidence_rate: float = 0.2,
        drop_feature_rate: float = 0.0,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
        graph_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.dataset = dataset
        self.graph_batch_size = graph_batch_size
        self.views_per_epoch = views_per_epoch
        self.drop_incidence_rate = drop_incidence_rate
        self.drop_feature_rate = drop_feature_rate
        self.shuffle = shuffle
        self.generator = generator
        self.graph_transform = graph_transform

    def __len__(self) -> int:
        return self.views_per_epoch * math.ceil(len(self.dataset) / self.graph_batch_size)

    def _augment_view(self, batch: Any) -> Any:
        hyperedge_index = _get_hyperedge_index(batch)
        if self.drop_incidence_rate > 0:
            keep = torch.rand(hyperedge_index.shape[1], generator=self.generator)
            keep = (keep >= self.drop_incidence_rate).to(hyperedge_index.device)
            if keep.sum() == 0:
                keep[torch.randint(keep.numel(), (1,), generator=self.generator).item()] = True
            hyperedge_index = hyperedge_index[:, keep]

        x = None
        if self.drop_feature_rate > 0:
            x = _get_x(batch).clone()
            mask = torch.rand(x.shape[0], generator=self.generator).to(x.device) < self.drop_feature_rate
            x[mask] = 0

        return _maybe_transform_graph(_clone_graph(batch, hyperedge_index, x=x), self.graph_transform)

    def __iter__(self) -> Iterator[TaskBatch]:
        for view_id in range(self.views_per_epoch):
            loader = DataLoader(self.dataset, batch_size=self.graph_batch_size, shuffle=self.shuffle)
            for graph_batch in loader:
                yield TaskBatch(
                    h_prime=(self._augment_view(graph_batch), self._augment_view(graph_batch)),
                    query=ContrastiveQuery(anchor_type="graph"),
                    task_type=self.task_type,
                    y=None,
                    split="pretrain",
                    metadata={"view_id": view_id},
                )
