from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Optional

import torch
from torch_geometric.data import Data

from .base import GraphQuery, TaskBatch, TaskType, normalize_split_name


SUBGRAPH_ROLE_DIM = 1


def is_subgraph_mode(args: Any) -> bool:
    return getattr(args, "subgraph_mode", "full") != "full"


def subgraph_role_dim(args: Any) -> int:
    if not bool(getattr(args, "subgraph_add_role_features", True)):
        return 0
    return int(getattr(args, "subgraph_role_dim", SUBGRAPH_ROLE_DIM))


def _get_hyperedge_index(data: Any) -> torch.Tensor:
    if hasattr(data, "hyperedge_index"):
        return data.hyperedge_index
    if hasattr(data, "edge_index"):
        return data.edge_index
    if hasattr(data, "data"):
        return _get_hyperedge_index(data.data)
    raise AttributeError("data must expose hyperedge_index or edge_index")


def _get_x(data: Any) -> torch.Tensor:
    if hasattr(data, "x") and isinstance(data.x, torch.Tensor):
        return data.x
    if hasattr(data, "data") and hasattr(data.data, "x"):
        return data.data.x
    raise AttributeError("data must expose x")


def model_data_with_subgraph_schema(data: Any, args: Any) -> Any:
    role_dim = subgraph_role_dim(args)
    if role_dim <= 0:
        return data

    model_data = copy.copy(data)
    if hasattr(data, "x") and isinstance(data.x, torch.Tensor):
        x = data.x
        model_data.x = x.new_zeros((max(int(x.shape[0]), 1), int(x.shape[1]) + role_dim))
        try:
            model_data.num_features = int(x.shape[1]) + role_dim
        except Exception:
            pass
        return model_data

    if hasattr(data, "num_features"):
        model_data.num_features = int(data.num_features) + role_dim
        return model_data

    return data


def append_subgraph_role_features(
    x: torch.Tensor,
    query_mask: torch.Tensor,
    task_type: TaskType | str,
    args: Any,
) -> torch.Tensor:
    role_dim = subgraph_role_dim(args)
    if role_dim <= 0:
        return x

    query_mask = query_mask.to(x.device).bool().view(-1)
    role = x.new_zeros((x.shape[0], role_dim))
    if role_dim >= 1:
        role[:, 0] = query_mask.to(dtype=x.dtype)
    return torch.cat([x, role], dim=-1)


def append_graph_batch_role_features(batch: Any, task_type: TaskType | str, args: Any) -> Any:
    if subgraph_role_dim(args) <= 0:
        return batch
    if not hasattr(batch, "x") or not isinstance(batch.x, torch.Tensor):
        return batch
    query_mask = torch.zeros(batch.x.shape[0], dtype=torch.bool, device=batch.x.device)
    batch.x = append_subgraph_role_features(batch.x, query_mask, task_type, args)
    return batch


def _batch_subgraphs(graphs: Sequence[Data]) -> Data:
    if not graphs:
        raise ValueError("Cannot batch an empty subgraph list")

    xs = []
    ys = []
    batch_vec = []
    query_masks = []
    hyperedge_parts = []
    node_offset = 0
    edge_offset = 0

    for graph_id, graph in enumerate(graphs):
        x = graph.x
        hyperedge_index = graph.hyperedge_index
        xs.append(x)
        ys.append(graph.y.view(-1))
        batch_vec.append(torch.full((x.shape[0],), graph_id, dtype=torch.long))
        if hasattr(graph, "query_mask"):
            query_masks.append(graph.query_mask.view(-1).bool())
        else:
            query_masks.append(torch.zeros(x.shape[0], dtype=torch.bool))

        if hyperedge_index.numel() > 0:
            shifted = hyperedge_index.clone()
            shifted[0] += node_offset
            shifted[1] += edge_offset
            hyperedge_parts.append(shifted)
            edge_offset += int(hyperedge_index[1].max().item()) + 1

        node_offset += int(x.shape[0])

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    batch = torch.cat(batch_vec, dim=0)
    query_mask = torch.cat(query_masks, dim=0)
    if hyperedge_parts:
        hyperedge_index = torch.cat(hyperedge_parts, dim=1)
    else:
        hyperedge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(
        x=x,
        hyperedge_index=hyperedge_index,
        edge_index=hyperedge_index,
        y=y,
        batch=batch,
        query_mask=query_mask,
        num_nodes=int(x.shape[0]),
        num_hyperedges=int(edge_offset),
    )


class PropagationSubgraphBuilder:
    def __init__(self, data: Any, args: Any):
        self.data = data
        self.args = args
        self.x = _get_x(data).detach().cpu()
        hyperedge_index = _get_hyperedge_index(data).detach().cpu().long()
        self.hyperedge_index = hyperedge_index
        self.context_hops = max(int(getattr(args, "subgraph_context_hops", 1)), 1)
        self.max_nodes = int(getattr(args, "subgraph_max_nodes", 0))
        self.max_hyperedges = int(getattr(args, "subgraph_max_hyperedges", 32))

        num_nodes = int(self.x.shape[0])
        num_edges = int(hyperedge_index[1].max().item()) + 1 if hyperedge_index.numel() else 0
        self.node_edges: list[list[int]] = [[] for _ in range(num_nodes)]
        self.edge_nodes: list[list[int]] = [[] for _ in range(num_edges)]
        for node_id, edge_id in hyperedge_index.t().tolist():
            if 0 <= int(node_id) < num_nodes and 0 <= int(edge_id) < num_edges:
                self.node_edges[int(node_id)].append(int(edge_id))
                self.edge_nodes[int(edge_id)].append(int(node_id))

        self.node_edges = [sorted(set(edges)) for edges in self.node_edges]
        self.edge_nodes = [sorted(set(nodes)) for nodes in self.edge_nodes]

    def _exact_edge_ids(self, nodes: Sequence[int]) -> set[int]:
        target = set(int(node) for node in nodes)
        if not target:
            return set()
        return {
            edge_id
            for edge_id, edge_nodes in enumerate(self.edge_nodes)
            if set(edge_nodes) == target
        }

    def _edge_sort_key(self, edge_id: int, seed_set: set[int], frontier: set[int]) -> tuple[int, int, int, int]:
        edge_node_set = set(self.edge_nodes[edge_id])
        seed_overlap = len(edge_node_set.intersection(seed_set))
        frontier_overlap = len(edge_node_set.intersection(frontier))
        edge_size = len(edge_node_set)
        return (-seed_overlap, -frontier_overlap, edge_size, edge_id)

    def _collect_ids(
        self,
        seed_nodes: Sequence[int],
        exclude_edge_ids: Optional[set[int]] = None,
    ) -> tuple[list[int], list[int]]:
        seed = sorted(set(int(node) for node in seed_nodes if 0 <= int(node) < self.x.shape[0]))
        if not seed:
            seed = [0]

        exclude_edge_ids = exclude_edge_ids or set()
        seed_set = set(seed)
        selected_node_set = set(seed)
        selected_edges: list[int] = []
        selected_edge_set: set[int] = set()
        frontier = set(seed)

        for _ in range(self.context_hops):
            incident_edges: set[int] = set()
            for node_id in frontier:
                incident_edges.update(self.node_edges[node_id])
            incident_edges.difference_update(exclude_edge_ids)
            incident_edges.difference_update(selected_edge_set)
            if not incident_edges:
                break

            ranked_edges = sorted(
                incident_edges,
                key=lambda edge_id: self._edge_sort_key(edge_id, seed_set, frontier),
            )
            if self.max_hyperedges > 0:
                remaining_edges = self.max_hyperedges - len(selected_edges)
                if remaining_edges <= 0:
                    break
                ranked_edges = ranked_edges[:remaining_edges]
            if not ranked_edges:
                break

            selected_edges.extend(ranked_edges)
            selected_edge_set.update(ranked_edges)

            member_nodes: set[int] = set()
            for edge_id in ranked_edges:
                member_nodes.update(self.edge_nodes[edge_id])

            new_nodes = member_nodes.difference(selected_node_set)
            if self.max_nodes > 0:
                remaining_nodes = self.max_nodes - len(selected_node_set)
                if remaining_nodes <= 0:
                    break
                new_nodes = set(sorted(new_nodes)[:remaining_nodes])

            selected_node_set.update(new_nodes)
            if not new_nodes:
                break
            frontier = new_nodes

        context_nodes = sorted(selected_node_set.difference(seed_set))
        selected_nodes = seed + context_nodes
        return selected_nodes, selected_edges

    def build_graph(
        self,
        seed_nodes: Sequence[int],
        y: int | torch.Tensor,
        task_type: TaskType | str,
        exclude_exact_seed_edge: bool = False,
    ) -> Data:
        seed = sorted(set(int(node) for node in seed_nodes if 0 <= int(node) < self.x.shape[0]))
        exclude = self._exact_edge_ids(seed) if exclude_exact_seed_edge else set()
        selected_nodes, selected_edges = self._collect_ids(seed, exclude)

        node_to_local = {node_id: idx for idx, node_id in enumerate(selected_nodes)}
        x = self.x[selected_nodes].clone()
        query_mask = torch.tensor([node_id in set(seed) for node_id in selected_nodes], dtype=torch.bool)
        x = append_subgraph_role_features(x, query_mask, task_type, self.args)

        rows = []
        cols = []
        local_edge_id = 0
        for edge_id in selected_edges:
            local_nodes = [node_to_local[node] for node in self.edge_nodes[edge_id] if node in node_to_local]
            if not local_nodes:
                continue
            rows.extend(local_nodes)
            cols.extend([local_edge_id] * len(local_nodes))
            local_edge_id += 1

        if not rows:
            for node_id in range(len(selected_nodes)):
                rows.append(node_id)
                cols.append(node_id)
            local_edge_id = len(selected_nodes)

        hyperedge_index = torch.tensor([rows, cols], dtype=torch.long)
        label = y.detach().cpu().view(-1).long() if isinstance(y, torch.Tensor) else torch.tensor([int(y)], dtype=torch.long)
        return Data(
            x=x,
            hyperedge_index=hyperedge_index,
            edge_index=hyperedge_index,
            y=label,
            query_mask=query_mask,
            num_nodes=int(x.shape[0]),
            num_hyperedges=int(local_edge_id),
        )

    def build_batch(
        self,
        seeds: Sequence[Sequence[int]],
        labels: Sequence[int] | torch.Tensor,
        task_type: TaskType | str,
        exclude_exact_seed_edge: bool = False,
    ) -> Data:
        if isinstance(labels, torch.Tensor):
            label_values = labels.detach().cpu().view(-1).tolist()
        else:
            label_values = list(labels)
        graphs = [
            self.build_graph(seed, label, task_type, exclude_exact_seed_edge=exclude_exact_seed_edge)
            for seed, label in zip(seeds, label_values)
        ]
        return _batch_subgraphs(graphs)


class NodeClsSubgraphTaskAdapter:
    task_type = TaskType.NODE_CLS

    def __init__(
        self,
        data: Any,
        masks: Mapping[str, torch.Tensor],
        args: Any,
        split: str = "train",
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        builder: Optional[PropagationSubgraphBuilder] = None,
        cache: Optional[bool] = None,
    ) -> None:
        self.data = data
        self.labels = data.y if hasattr(data, "y") else data.data.y
        self.masks = masks
        self.args = args
        self.split = normalize_split_name(split)
        self.batch_size = batch_size or int(getattr(args, "subgraph_batch_size", 128))
        self.shuffle = shuffle
        self.builder = builder or PropagationSubgraphBuilder(data, args)
        self.cache = bool(getattr(args, "subgraph_cache", True)) if cache is None else bool(cache)
        self.node_ids = self._node_ids().detach().cpu()
        self.graphs = self._build_cached_graphs() if self.cache else None
        self.batches = self._build_cached_batches() if self.graphs is not None else None

    def _node_ids(self) -> torch.Tensor:
        mask_or_idx = self.masks[self.split]
        if mask_or_idx.dtype == torch.bool:
            return mask_or_idx.nonzero(as_tuple=False).view(-1)
        return mask_or_idx.view(-1).long()

    def _build_cached_graphs(self) -> list[Data]:
        if self.node_ids.numel() == 0:
            return []

        labels = self.labels.detach().cpu()[self.node_ids]
        return [
            self.builder.build_graph(
                [int(node_id)],
                labels[offset],
                self.task_type,
                exclude_exact_seed_edge=False,
            )
            for offset, node_id in enumerate(self.node_ids.tolist())
        ]

    def __len__(self) -> int:
        count = int(self.node_ids.numel())
        if count == 0:
            return 0
        return (count + self.batch_size - 1) // self.batch_size

    def _build_cached_batches(self) -> list[tuple[torch.Tensor, Data]]:
        batches = []
        for start in range(0, self.node_ids.numel(), self.batch_size):
            batch_ids = self.node_ids[start : start + self.batch_size]
            graphs = self.graphs[start : start + self.batch_size]
            batches.append((batch_ids, _batch_subgraphs(graphs)))
        return batches

    def __iter__(self) -> Iterator[TaskBatch]:
        node_ids = self.node_ids
        if node_ids.numel() == 0:
            return

        if self.batches is not None:
            batch_order = torch.arange(len(self.batches))
            if self.shuffle:
                batch_order = batch_order[torch.randperm(batch_order.numel())]

            for batch_idx in batch_order.tolist():
                batch_ids, subgraph_batch = self.batches[int(batch_idx)]
                graph_ids = torch.arange(int(subgraph_batch.y.numel()), dtype=torch.long)
                yield TaskBatch(
                    h_prime=subgraph_batch,
                    query=GraphQuery(graph_ids),
                    task_type=self.task_type,
                    y=subgraph_batch.y,
                    split=self.split,
                    metadata={"node_ids": batch_ids},
                )
            return

        order = torch.arange(node_ids.numel())
        if self.shuffle:
            order = order[torch.randperm(order.numel())]

        for start in range(0, order.numel(), self.batch_size):
            batch_order = order[start : start + self.batch_size]
            batch_ids = node_ids[batch_order]

            if self.graphs is not None:
                subgraph_batch = _batch_subgraphs([self.graphs[int(idx)] for idx in batch_order.tolist()])
            else:
                labels = self.labels.detach().cpu()[batch_ids]
                subgraph_batch = self.builder.build_batch(
                    [[int(node_id)] for node_id in batch_ids.tolist()],
                    labels,
                    self.task_type,
                    exclude_exact_seed_edge=False,
                )

            graph_ids = torch.arange(int(subgraph_batch.y.numel()), dtype=torch.long)
            yield TaskBatch(
                h_prime=subgraph_batch,
                query=GraphQuery(graph_ids),
                task_type=self.task_type,
                y=subgraph_batch.y,
                split=self.split,
                metadata={"node_ids": batch_ids},
            )


def build_edge_subgraph_task_batch(
    builder: PropagationSubgraphBuilder,
    hyperedges: Sequence[torch.Tensor | Sequence[int]],
    labels: torch.Tensor,
) -> TaskBatch:
    seeds = []
    for hyperedge in hyperedges:
        if isinstance(hyperedge, torch.Tensor):
            seeds.append([int(node) for node in hyperedge.detach().cpu().view(-1).tolist()])
        else:
            seeds.append([int(node) for node in hyperedge])

    label_cpu = labels.detach().cpu().view(-1).long()
    subgraph_batch = builder.build_batch(
        seeds,
        label_cpu,
        TaskType.EDGE_PRED,
        exclude_exact_seed_edge=True,
    )
    graph_ids = torch.arange(int(label_cpu.numel()), dtype=torch.long)
    return TaskBatch(
        h_prime=subgraph_batch,
        query=GraphQuery(graph_ids),
        task_type=TaskType.EDGE_PRED,
        y=label_cpu,
        split="edge_pred",
    )
