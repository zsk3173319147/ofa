from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Optional

import torch
from torch_geometric.data import Data

from .base import GraphQuery, TaskBatch, TaskType, normalize_split_name


SUBGRAPH_ROLE_DIM = 3


def is_subgraph_mode(args: Any) -> bool:
    return getattr(args, "subgraph_mode", "full") != "full"


def subgraph_role_dim(args: Any) -> int:
    if not bool(getattr(args, "subgraph_add_role_features", True)):
        return 0
    return int(getattr(args, "subgraph_role_dim", SUBGRAPH_ROLE_DIM))


def task_scalar(task_type: TaskType | str) -> float:
    task_type = TaskType(task_type)
    if task_type == TaskType.NODE_CLS:
        return 0.0
    if task_type == TaskType.EDGE_PRED:
        return 0.5
    if task_type == TaskType.HG_CLS:
        return 1.0
    return 0.0


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
    if role_dim >= 2:
        role[:, 1] = (~query_mask).to(dtype=x.dtype)
    if role_dim >= 3:
        role[:, 2] = task_scalar(task_type)
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
    hyperedge_parts = []
    node_offset = 0
    edge_offset = 0

    for graph_id, graph in enumerate(graphs):
        x = graph.x
        hyperedge_index = graph.hyperedge_index
        xs.append(x)
        ys.append(graph.y.view(-1))
        batch_vec.append(torch.full((x.shape[0],), graph_id, dtype=torch.long))

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
        self.max_nodes = int(getattr(args, "subgraph_max_nodes", 64))
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

    def _collect_ids(
        self,
        seed_nodes: Sequence[int],
        exclude_edge_ids: Optional[set[int]] = None,
    ) -> tuple[list[int], list[int]]:
        seed = sorted(set(int(node) for node in seed_nodes if 0 <= int(node) < self.x.shape[0]))
        if not seed:
            seed = [0]

        exclude_edge_ids = exclude_edge_ids or set()
        visited_nodes = set(seed)
        visited_edges: set[int] = set()
        frontier = set(seed)

        for _ in range(self.context_hops):
            incident_edges: set[int] = set()
            for node_id in frontier:
                incident_edges.update(self.node_edges[node_id])
            incident_edges.difference_update(exclude_edge_ids)
            if not incident_edges:
                break

            visited_edges.update(incident_edges)
            member_nodes: set[int] = set()
            for edge_id in incident_edges:
                member_nodes.update(self.edge_nodes[edge_id])
            new_nodes = member_nodes.difference(visited_nodes)
            visited_nodes.update(member_nodes)
            if not new_nodes:
                break
            frontier = new_nodes

        context_nodes = sorted(visited_nodes.difference(seed))
        if self.max_nodes > 0:
            context_budget = max(self.max_nodes - len(seed), 0)
            context_nodes = context_nodes[:context_budget]
        selected_nodes = seed + context_nodes

        selected_node_set = set(selected_nodes)
        selected_edges = [
            edge_id
            for edge_id in sorted(visited_edges)
            if any(node in selected_node_set for node in self.edge_nodes[edge_id])
        ]
        if self.max_hyperedges > 0:
            selected_edges = selected_edges[: self.max_hyperedges]
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
    ) -> None:
        self.data = data
        self.labels = data.y if hasattr(data, "y") else data.data.y
        self.masks = masks
        self.args = args
        self.split = normalize_split_name(split)
        self.batch_size = batch_size or int(getattr(args, "subgraph_batch_size", 128))
        self.shuffle = shuffle
        self.builder = PropagationSubgraphBuilder(data, args)

    def _node_ids(self) -> torch.Tensor:
        mask_or_idx = self.masks[self.split]
        if mask_or_idx.dtype == torch.bool:
            return mask_or_idx.nonzero(as_tuple=False).view(-1)
        return mask_or_idx.view(-1).long()

    def __iter__(self) -> Iterator[TaskBatch]:
        node_ids = self._node_ids().detach().cpu()
        if node_ids.numel() == 0:
            return
        if self.shuffle:
            node_ids = node_ids[torch.randperm(node_ids.numel())]

        for start in range(0, node_ids.numel(), self.batch_size):
            batch_ids = node_ids[start : start + self.batch_size]
            labels = self.labels.detach().cpu()[batch_ids]
            subgraph_batch = self.builder.build_batch(
                [[int(node_id)] for node_id in batch_ids.tolist()],
                labels,
                self.task_type,
                exclude_exact_seed_edge=False,
            )
            graph_ids = torch.arange(int(labels.numel()), dtype=torch.long)
            yield TaskBatch(
                h_prime=subgraph_batch,
                query=GraphQuery(graph_ids),
                task_type=self.task_type,
                y=labels,
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
