from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from lib_models.HNN import MLP
from tasker import GraphQuery, HyperedgeQuery, NodeQuery, TaskType


TASK_ID = {
    TaskType.NODE_CLS: 0,
    TaskType.EDGE_PRED: 1,
    TaskType.HG_CLS: 2,
}

ROLE_QUERY = 0
ROLE_TASK = 1
ROLE_OBJECT = 2
ROLE_CONTEXT = 3
ROLE_HYPEREDGE_CONTEXT = 4


def split_encoder_output(output: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(output, (tuple, list)):
        node_emb = output[0]
        edge_emb = output[1] if len(output) > 1 else None
        return node_emb, edge_emb
    return output, None


def reset_dynamic_encoder_state(encoder: nn.Module) -> None:
    for attr in ("cache", "structure", "hyperedge_attr"):
        if hasattr(encoder, attr):
            setattr(encoder, attr, None)


def mean_pool_by_index(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.numel() == 0:
        return values.mean(dim=0, keepdim=True)

    dim_size = int(index.max().item()) + 1
    pooled = values.new_zeros(dim_size, values.shape[-1])
    counts = values.new_zeros(dim_size, 1)
    pooled.index_add_(0, index, values)
    counts.index_add_(0, index, torch.ones(index.shape[0], 1, dtype=values.dtype, device=values.device))
    return pooled / counts.clamp_min(1)


def get_hyperedge_index(data: Any) -> torch.Tensor:
    if hasattr(data, "hyperedge_index"):
        return data.hyperedge_index
    if hasattr(data, "edge_index"):
        return data.edge_index
    if hasattr(data, "data"):
        return get_hyperedge_index(data.data)
    raise AttributeError("data must expose hyperedge_index, edge_index, or data.hyperedge_index")


def get_num_nodes(data: Any, hyperedge_index: torch.Tensor) -> int:
    if hasattr(data, "x"):
        return int(data.x.shape[0])
    if hasattr(data, "data") and hasattr(data.data, "x"):
        return int(data.data.x.shape[0])
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        num_nodes = data.num_nodes
        return int(num_nodes.item()) if isinstance(num_nodes, torch.Tensor) else int(num_nodes)
    if hyperedge_index.numel() == 0:
        return 0
    return int(hyperedge_index[0].max().item()) + 1


class TargetAwareReadout(nn.Module):
    """Builds one query representation from the object named by TaskBatch.query."""

    def __init__(self, args):
        super().__init__()
        self.edge_aggr = getattr(args, "aggr_mode", "max")
        self.graph_pooling = getattr(args, "pooling", "mean")

    def reset_parameters(self) -> None:
        return

    def _pool_node_set(self, node_emb: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        node_ids = node_ids.to(node_emb.device).long().view(-1)
        if node_ids.numel() == 0:
            return node_emb.new_zeros(node_emb.shape[-1])

        values = node_emb[node_ids]
        if self.edge_aggr == "mean":
            return values.mean(dim=0).abs()
        if self.edge_aggr == "maxmin":
            return values.max(dim=0).values - values.min(dim=0).values
        return values.max(dim=0).values

    def _read_hyperedges(self, node_emb: torch.Tensor, query: HyperedgeQuery) -> torch.Tensor:
        return torch.stack([self._pool_node_set(node_emb, hyperedge) for hyperedge in query.hyperedges], dim=0)

    def _read_graphs(self, node_emb: torch.Tensor, data: Any, query: GraphQuery) -> torch.Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch.to(node_emb.device).long()
            if self.graph_pooling == "max":
                graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
                pooled = []
                for graph_id in range(graph_count):
                    values = node_emb[batch == graph_id]
                    if values.numel() == 0:
                        pooled.append(node_emb.new_zeros(node_emb.shape[-1]))
                    else:
                        pooled.append(values.max(dim=0).values)
                graph_emb = torch.stack(pooled, dim=0)
            else:
                graph_emb = mean_pool_by_index(node_emb, batch)
        else:
            graph_emb = node_emb.mean(dim=0, keepdim=True)

        if query.graph_ids is None:
            return graph_emb
        return graph_emb[query.graph_ids.to(graph_emb.device).long()]

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: Optional[torch.Tensor],
        data: Any,
        query: NodeQuery | HyperedgeQuery | GraphQuery,
        task_type: TaskType,
    ) -> torch.Tensor:
        if isinstance(query, NodeQuery):
            return node_emb[query.node_ids.to(node_emb.device).long()]
        if isinstance(query, HyperedgeQuery):
            return self._read_hyperedges(node_emb, query)
        if isinstance(query, GraphQuery):
            return self._read_graphs(node_emb, data, query)
        raise ValueError(f"Unsupported query type for unified downstream task: {type(query).__name__}")


class TaskHypergraphLayer(nn.Module):
    """A small hypergraph propagation layer used only on task-side tokens."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        self.node_proj.reset_parameters()
        self.msg_proj.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, tokens: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        if incidence.numel() == 0:
            return tokens

        node_ids = incidence[0].long()
        edge_ids = incidence[1].long()
        hidden = self.node_proj(tokens)

        num_edges = int(edge_ids.max().item()) + 1
        edge_emb = hidden.new_zeros(num_edges, hidden.shape[-1])
        edge_count = hidden.new_zeros(num_edges, 1)
        edge_emb.index_add_(0, edge_ids, hidden[node_ids])
        edge_count.index_add_(0, edge_ids, torch.ones(edge_ids.shape[0], 1, dtype=hidden.dtype, device=hidden.device))
        edge_emb = edge_emb / edge_count.clamp_min(1)

        node_msg = hidden.new_zeros(tokens.shape[0], hidden.shape[-1])
        node_count = hidden.new_zeros(tokens.shape[0], 1)
        node_msg.index_add_(0, node_ids, edge_emb[edge_ids])
        node_count.index_add_(0, node_ids, torch.ones(node_ids.shape[0], 1, dtype=hidden.dtype, device=hidden.device))
        node_msg = node_msg / node_count.clamp_min(1)

        update = self.dropout(self.activation(self.msg_proj(node_msg)))
        return self.norm(tokens + update)


class TaskHypergraphReasoner(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            TaskHypergraphLayer(hidden_dim, dropout=dropout)
            for _ in range(max(int(num_layers), 1))
        )

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, tokens: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, incidence)
        return tokens


class TaskHypergraphReadout(nn.Module):
    """
    Decoupled task-side hypergraph readout.

    The original hypergraph is encoded first. This module then builds a small
    task hypergraph over encoded object/context representations and reads Q.
    """

    def __init__(self, args):
        super().__init__()
        self.embedding_dim = args.embedding_hidden
        self.edge_aggr = getattr(args, "aggr_mode", "max")
        self.graph_pooling = getattr(args, "pooling", "mean")
        context_hops = int(getattr(args, "task_context_hops", 1))
        if context_hops <= 0:
            context_hops = int(getattr(args, "All_num_layers", 1))
        self.context_hops = max(context_hops, 1)
        self.max_node_context = getattr(args, "task_max_node_context", 32)
        self.max_hyperedge_context = getattr(args, "task_max_hyperedge_context", 16)
        self.query_embeddings = nn.Embedding(len(TASK_ID), self.embedding_dim)
        self.task_embeddings = nn.Embedding(len(TASK_ID), self.embedding_dim)
        self.role_embeddings = nn.Embedding(5, self.embedding_dim)
        self.reasoner = TaskHypergraphReasoner(
            self.embedding_dim,
            num_layers=getattr(args, "task_reason_layers", 1),
            dropout=getattr(args, "task_reason_dropout", 0.0),
        )
        self._context_cache: dict[tuple[int, str, int], dict[str, list[torch.Tensor]]] = {}

    def reset_parameters(self) -> None:
        self.query_embeddings.reset_parameters()
        self.task_embeddings.reset_parameters()
        self.role_embeddings.reset_parameters()
        self.reasoner.reset_parameters()

    def _task_id(self, task_type: TaskType) -> int:
        return TASK_ID[TaskType(task_type)]

    def _role(self, role_id: int, device: torch.device) -> torch.Tensor:
        return self.role_embeddings(torch.tensor(role_id, dtype=torch.long, device=device))

    def _pool_values(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_zeros(self.embedding_dim)
        if self.edge_aggr == "mean":
            return values.mean(dim=0).abs()
        if self.edge_aggr == "maxmin":
            return values.max(dim=0).values - values.min(dim=0).values
        return values.max(dim=0).values

    def _limit_ids(self, ids: torch.Tensor, max_count: int | None) -> torch.Tensor:
        if max_count is None or max_count <= 0 or ids.numel() <= max_count:
            return ids
        return ids[:max_count]

    def _isin(self, values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.numel() == 0:
            return torch.zeros_like(values, dtype=torch.bool)
        if hasattr(torch, "isin"):
            return torch.isin(values, candidates)
        mask = torch.zeros_like(values, dtype=torch.bool)
        for item in candidates:
            mask |= values == item
        return mask

    def _incidence_cache(self, data: Any, device: torch.device) -> dict[str, list[torch.Tensor]]:
        hyperedge_index = get_hyperedge_index(data).to(device)
        key = (id(data), str(device), int(hyperedge_index.shape[1]))
        cached = self._context_cache.get(key)
        if cached is not None:
            return cached

        num_nodes = get_num_nodes(data, hyperedge_index)
        num_edges = int(hyperedge_index[1].max().item()) + 1 if hyperedge_index.numel() else 0
        node_edges: list[list[int]] = [[] for _ in range(num_nodes)]
        edge_nodes: list[list[int]] = [[] for _ in range(num_edges)]

        for node_id, edge_id in hyperedge_index.detach().cpu().t().tolist():
            node_id = int(node_id)
            edge_id = int(edge_id)
            if 0 <= node_id < num_nodes:
                node_edges[node_id].append(edge_id)
            if 0 <= edge_id < num_edges:
                edge_nodes[edge_id].append(node_id)

        cache = {
            "node_edges": [
                torch.tensor(sorted(set(edges)), dtype=torch.long, device=device)
                for edges in node_edges
            ],
            "edge_nodes": [
                torch.tensor(sorted(set(nodes)), dtype=torch.long, device=device)
                for nodes in edge_nodes
            ],
        }
        self._context_cache[key] = cache
        return cache

    def _incident_context_ids(
        self,
        data: Any,
        seed_nodes: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seed_nodes.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        seed_nodes = torch.unique(seed_nodes.to(device).long().view(-1))
        cache = self._incidence_cache(data, device)
        node_edges = cache["node_edges"]
        edge_nodes = cache["edge_nodes"]

        visited_nodes = seed_nodes
        frontier_nodes = seed_nodes
        visited_edges = torch.empty(0, dtype=torch.long, device=device)

        for _ in range(self.context_hops):
            edge_parts = []
            for node_id in frontier_nodes.detach().cpu().tolist():
                if 0 <= int(node_id) < len(node_edges):
                    edge_parts.append(node_edges[int(node_id)])
            if not edge_parts:
                break

            incident_edges = torch.unique(torch.cat(edge_parts, dim=0))
            if incident_edges.numel() == 0:
                break
            visited_edges = torch.unique(torch.cat([visited_edges, incident_edges], dim=0))

            node_parts = []
            for edge_id in incident_edges.detach().cpu().tolist():
                if 0 <= int(edge_id) < len(edge_nodes):
                    node_parts.append(edge_nodes[int(edge_id)])
            if not node_parts:
                break

            member_nodes = torch.unique(torch.cat(node_parts, dim=0))
            new_nodes = member_nodes[~self._isin(member_nodes, visited_nodes)]
            visited_nodes = torch.unique(torch.cat([visited_nodes, member_nodes], dim=0))
            if new_nodes.numel() == 0:
                break
            frontier_nodes = new_nodes

        if visited_edges.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        node_context = visited_nodes[~self._isin(visited_nodes, seed_nodes)]

        node_context = self._limit_ids(node_context, self.max_node_context)
        visited_edges = self._limit_ids(visited_edges, self.max_hyperedge_context)
        return node_context, visited_edges

    def _edge_context_tokens(
        self,
        node_emb: torch.Tensor,
        edge_emb: Optional[torch.Tensor],
        data: Any,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        edge_ids = edge_ids.to(node_emb.device).long().view(-1)
        if edge_ids.numel() == 0:
            return node_emb.new_empty(0, node_emb.shape[-1])

        if edge_emb is not None and edge_emb.shape[0] > int(edge_ids.max().item()):
            return edge_emb.to(node_emb.device)[edge_ids]

        hyperedge_index = get_hyperedge_index(data).to(node_emb.device)
        edge_col = hyperedge_index[1].long()
        tokens = []
        for edge_id in edge_ids:
            nodes = hyperedge_index[0, edge_col == edge_id].long()
            if nodes.numel() == 0:
                tokens.append(node_emb.new_zeros(node_emb.shape[-1]))
            else:
                tokens.append(node_emb[nodes].mean(dim=0))
        return torch.stack(tokens, dim=0)

    def _graph_context_ids(
        self,
        data: Any,
        graph_id: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hyperedge_index = get_hyperedge_index(data).to(device)
        if hasattr(data, "batch") and data.batch is not None:
            graph_nodes = torch.where(data.batch.to(device).long() == graph_id.long())[0]
        else:
            graph_nodes = torch.arange(get_num_nodes(data, hyperedge_index), dtype=torch.long, device=device)

        if graph_nodes.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        node_context_ids = self._limit_ids(graph_nodes, self.max_node_context)
        _, edge_context_ids = self._incident_context_ids(data, graph_nodes, device)
        return node_context_ids, edge_context_ids

    def _graph_embeddings(self, node_emb: torch.Tensor, data: Any) -> torch.Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch.to(node_emb.device).long()
            if self.graph_pooling == "max":
                graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
                pooled = []
                for graph_id in range(graph_count):
                    values = node_emb[batch == graph_id]
                    pooled.append(values.max(dim=0).values if values.numel() else node_emb.new_zeros(node_emb.shape[-1]))
                return torch.stack(pooled, dim=0)
            return mean_pool_by_index(node_emb, batch)
        return node_emb.mean(dim=0, keepdim=True)

    def _incidence(
        self,
        object_ids: list[int],
        node_context_ids: list[int],
        edge_context_ids: list[int],
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Token layout: 0=Q, 1=T, followed by object/node-context/hyperedge-context tokens.
        edges: list[tuple[int, int]] = []

        # e_task = {Q, T}
        edges.extend([(0, 0), (1, 0)])

        # e_obj = {Q} union O
        for token_id in [0] + object_ids:
            edges.append((token_id, 1))

        edge_offset = 2
        if node_context_ids:
            # e_ctx^n = {Q} union O union node-context tokens
            for token_id in [0] + object_ids + node_context_ids:
                edges.append((token_id, edge_offset))
            edge_offset += 1

        if edge_context_ids:
            # e_ctx^h = {Q} union O union hyperedge-context tokens
            for token_id in [0] + object_ids + edge_context_ids:
                edges.append((token_id, edge_offset))

        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    def _run_task_graph(
        self,
        object_tokens: torch.Tensor,
        node_context_tokens: Optional[torch.Tensor],
        edge_context_tokens: Optional[torch.Tensor],
        task_type: TaskType,
    ) -> torch.Tensor:
        device = object_tokens.device
        task_id = torch.tensor(self._task_id(task_type), dtype=torch.long, device=device)
        query_token = self.query_embeddings(task_id) + self._role(ROLE_QUERY, device)
        task_token = self.task_embeddings(task_id) + self._role(ROLE_TASK, device)
        object_tokens = object_tokens + self._role(ROLE_OBJECT, device)

        tokens = [query_token.view(1, -1), task_token.view(1, -1), object_tokens]
        object_ids = list(range(2, 2 + object_tokens.shape[0]))
        next_token_id = 2 + object_tokens.shape[0]

        node_context_ids: list[int] = []
        if node_context_tokens is not None and node_context_tokens.numel() > 0:
            node_context_tokens = node_context_tokens + self._role(ROLE_CONTEXT, device)
            node_context_ids = list(range(next_token_id, next_token_id + node_context_tokens.shape[0]))
            next_token_id += node_context_tokens.shape[0]
            tokens.append(node_context_tokens)

        edge_context_ids: list[int] = []
        if edge_context_tokens is not None and edge_context_tokens.numel() > 0:
            edge_context_tokens = edge_context_tokens + self._role(ROLE_HYPEREDGE_CONTEXT, device)
            edge_context_ids = list(range(next_token_id, next_token_id + edge_context_tokens.shape[0]))
            tokens.append(edge_context_tokens)

        task_tokens = torch.cat(tokens, dim=0)
        incidence = self._incidence(
            object_ids=object_ids,
            node_context_ids=node_context_ids,
            edge_context_ids=edge_context_ids,
            num_tokens=task_tokens.shape[0],
            device=device,
        )
        task_tokens = self.reasoner(task_tokens, incidence)
        return task_tokens[0]

    def _read_nodes(
        self,
        node_emb: torch.Tensor,
        edge_emb: Optional[torch.Tensor],
        data: Any,
        query: NodeQuery,
        task_type: TaskType,
    ) -> torch.Tensor:
        outputs = []
        for node_id in query.node_ids.to(node_emb.device).long().view(-1):
            node_context_ids, edge_context_ids = self._incident_context_ids(
                data,
                node_id.view(1),
                node_emb.device,
            )
            object_tokens = node_emb[node_id].view(1, -1)
            node_context_tokens = node_emb[node_context_ids] if node_context_ids.numel() else None
            edge_context_tokens = self._edge_context_tokens(node_emb, edge_emb, data, edge_context_ids)
            outputs.append(self._run_task_graph(object_tokens, node_context_tokens, edge_context_tokens, task_type))
        return torch.stack(outputs, dim=0)

    def _read_hyperedges(
        self,
        node_emb: torch.Tensor,
        edge_emb: Optional[torch.Tensor],
        data: Any,
        query: HyperedgeQuery,
        task_type: TaskType,
    ) -> torch.Tensor:
        outputs = []
        for hyperedge in query.hyperedges:
            node_ids = torch.unique(hyperedge.to(node_emb.device).long().view(-1))
            if node_ids.numel():
                node_context_ids, edge_context_ids = self._incident_context_ids(data, node_ids, node_emb.device)
                object_tokens = node_emb[node_ids]
                node_context_tokens = node_emb[node_context_ids] if node_context_ids.numel() else None
                edge_context_tokens = self._edge_context_tokens(node_emb, edge_emb, data, edge_context_ids)
            else:
                graph_context = node_emb.mean(dim=0, keepdim=True)
                object_tokens = graph_context
                node_context_tokens = None
                edge_context_tokens = None
            outputs.append(self._run_task_graph(object_tokens, node_context_tokens, edge_context_tokens, task_type))
        return torch.stack(outputs, dim=0)

    def _read_graphs(
        self,
        node_emb: torch.Tensor,
        edge_emb: Optional[torch.Tensor],
        data: Any,
        query: GraphQuery,
        task_type: TaskType,
    ) -> torch.Tensor:
        graph_emb = self._graph_embeddings(node_emb, data)
        if query.graph_ids is None:
            graph_ids = torch.arange(graph_emb.shape[0], dtype=torch.long, device=graph_emb.device)
        else:
            graph_ids = query.graph_ids.to(graph_emb.device).long().view(-1)

        if query.graph_ids is not None:
            graph_emb = graph_emb[graph_ids]

        outputs = []
        for object_token, graph_id in zip(graph_emb, graph_ids):
            node_context_ids, edge_context_ids = self._graph_context_ids(data, graph_id, node_emb.device)
            node_context_tokens = node_emb[node_context_ids] if node_context_ids.numel() else None
            edge_context_tokens = self._edge_context_tokens(node_emb, edge_emb, data, edge_context_ids)
            outputs.append(self._run_task_graph(object_token.view(1, -1), node_context_tokens, edge_context_tokens, task_type))
        return torch.stack(outputs, dim=0)

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: Optional[torch.Tensor],
        data: Any,
        query: NodeQuery | HyperedgeQuery | GraphQuery,
        task_type: TaskType,
    ) -> torch.Tensor:
        if isinstance(query, NodeQuery):
            return self._read_nodes(node_emb, edge_emb, data, query, task_type)
        if isinstance(query, HyperedgeQuery):
            return self._read_hyperedges(node_emb, edge_emb, data, query, task_type)
        if isinstance(query, GraphQuery):
            return self._read_graphs(node_emb, edge_emb, data, query, task_type)
        raise ValueError(f"Unsupported query type for task hypergraph readout: {type(query).__name__}")


class PromptHypergraphReadout(TaskHypergraphReadout):
    """
    Lightweight prompt-hypergraph readout.

    It reuses the same task object/context extraction as TaskHypergraphReadout,
    but compresses variable-size context into a fixed schema and applies a
    query-conditioned residual update to the object representation.
    """

    def __init__(self, args):
        nn.Module.__init__(self)
        self.embedding_dim = args.embedding_hidden
        self.edge_aggr = getattr(args, "aggr_mode", "max")
        self.graph_pooling = getattr(args, "pooling", "mean")
        context_hops = int(getattr(args, "task_context_hops", 1))
        if context_hops <= 0:
            context_hops = int(getattr(args, "All_num_layers", 1))
        self.context_hops = max(context_hops, 1)
        self.max_node_context = getattr(args, "task_max_node_context", 32)
        self.max_hyperedge_context = getattr(args, "task_max_hyperedge_context", 16)
        self.dropout = nn.Dropout(getattr(args, "prompt_readout_dropout", 0.0))
        self.residual_init = float(getattr(args, "prompt_residual_init", -5.0))

        self.query_embeddings = nn.Embedding(len(TASK_ID), self.embedding_dim)
        self.task_embeddings = nn.Embedding(len(TASK_ID), self.embedding_dim)
        self.role_embeddings = nn.Embedding(5, self.embedding_dim)
        self.query_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.delta_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(getattr(args, "prompt_readout_dropout", 0.0)),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.gate = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        self.residual_logit = nn.Parameter(torch.tensor(self.residual_init))
        self._context_cache: dict[tuple[int, str, int], dict[str, list[torch.Tensor]]] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.query_embeddings.reset_parameters()
        self.task_embeddings.reset_parameters()
        self.role_embeddings.reset_parameters()
        self.query_proj.reset_parameters()
        self.key_proj.reset_parameters()
        self.value_proj.reset_parameters()
        for module in self.delta_mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.gate.reset_parameters()
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        with torch.no_grad():
            self.residual_logit.fill_(self.residual_init)

    def _optional_context_summary(self, tokens: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        if tokens is None or tokens.numel() == 0:
            return None
        return self._pool_values(tokens).view(1, -1)

    def _pool_object_tokens(self, object_tokens: torch.Tensor) -> torch.Tensor:
        if object_tokens.shape[0] == 1:
            return object_tokens.view(1, -1)
        return self._pool_values(object_tokens).view(1, -1)

    def _run_task_graph(
        self,
        object_tokens: torch.Tensor,
        node_context_tokens: Optional[torch.Tensor],
        edge_context_tokens: Optional[torch.Tensor],
        task_type: TaskType,
    ) -> torch.Tensor:
        device = object_tokens.device
        task_id = torch.tensor(self._task_id(task_type), dtype=torch.long, device=device)
        object_summary = self._pool_object_tokens(object_tokens)
        query_token = self.query_embeddings(task_id).view(1, -1) + self._role(ROLE_QUERY, device).view(1, -1)
        task_token = self.task_embeddings(task_id).view(1, -1) + self._role(ROLE_TASK, device).view(1, -1)

        prompt_tokens = [
            object_summary + self._role(ROLE_OBJECT, device).view(1, -1),
            task_token,
        ]

        node_context = self._optional_context_summary(node_context_tokens, device)
        if node_context is not None:
            prompt_tokens.append(node_context + self._role(ROLE_CONTEXT, device).view(1, -1))

        edge_context = self._optional_context_summary(edge_context_tokens, device)
        if edge_context is not None:
            prompt_tokens.append(edge_context + self._role(ROLE_HYPEREDGE_CONTEXT, device).view(1, -1))

        prompt_tokens = torch.cat(prompt_tokens, dim=0)
        query = self.query_proj(query_token + object_summary)
        keys = self.key_proj(prompt_tokens)
        values = self.value_proj(prompt_tokens)
        scores = (keys * query).sum(dim=-1) / math.sqrt(self.embedding_dim)
        weights = torch.softmax(scores, dim=0).view(-1, 1)
        prompt_summary = (weights * values).sum(dim=0, keepdim=True)

        fused = torch.cat([object_summary, prompt_summary, query_token], dim=-1)
        delta = self.delta_mlp(self.dropout(fused))
        gate = torch.sigmoid(self.gate(fused))
        scale = torch.sigmoid(self.residual_logit)
        return (object_summary + scale * gate * delta).view(-1)


class UnifiedDownstreamModel(nn.Module):
    def __init__(self, encoder: nn.Module, task_type: TaskType | str, num_targets: int, args):
        super().__init__()
        self.encoder = encoder
        self.task_type = TaskType(task_type)
        self.readout = self._build_readout(args)
        self.head = self._build_head(num_targets, args)

    def _build_readout(self, args) -> nn.Module:
        readout = getattr(args, "unified_readout", "object")
        if readout == "object":
            return TargetAwareReadout(args)
        if readout == "task_hypergraph":
            return TaskHypergraphReadout(args)
        if readout == "prompt_hypergraph":
            return PromptHypergraphReadout(args)
        raise ValueError(f"Unsupported unified_readout: {readout}")

    def _head_type(self, args) -> str:
        head_type = getattr(args, "unified_head_type", "auto")
        if head_type != "auto":
            return head_type
        if self.task_type == TaskType.NODE_CLS:
            return "linear"
        return "mlp"

    def _build_head(self, num_targets: int, args) -> nn.Module:
        in_channels = args.embedding_hidden
        head_type = self._head_type(args)
        if head_type == "linear":
            return nn.Linear(in_channels, num_targets)

        if self.task_type == TaskType.EDGE_PRED:
            hidden = args.e_embed_hidden
            layers = args.e_embed_layer
            dropout = args.e_embed_dropout
            norm = args.e_embed_norm
        elif self.task_type == TaskType.HG_CLS:
            hidden = args.g_embed_hidden
            layers = args.g_embed_layer
            dropout = args.g_embed_dropout
            norm = args.g_embed_norm
        else:
            hidden = args.embedding_hidden
            layers = 2
            dropout = args.dropout
            norm = "ln"

        return MLP(
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=num_targets,
            num_layers=layers,
            dropout=dropout,
            Normalization=norm,
            InputNorm=False,
        )

    def reset_parameters(self) -> None:
        if hasattr(self.encoder, "reset_parameters"):
            self.encoder.reset_parameters()
        self.readout.reset_parameters()
        if hasattr(self.head, "reset_parameters"):
            self.head.reset_parameters()

    def encode(self, data: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        reset_dynamic_encoder_state(self.encoder)
        return split_encoder_output(self.encoder(data))

    def forward(self, batch) -> torch.Tensor:
        node_emb, edge_emb = self.encode(batch.h_prime)
        h_query = self.readout(node_emb, edge_emb, batch.h_prime, batch.query, batch.task_type)
        out = self.head(h_query)
        if batch.task_type == TaskType.EDGE_PRED:
            return out.view(-1)
        return out
