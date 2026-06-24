from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].item())
    return int(value)


class LearnableHyperedgePromptBank(nn.Module):
    """Query-conditioned structural prompt for sub-hypergraphs.

    For each graph in a batch, append K prompt nodes and K prompt hyperedges.
    Each prompt hyperedge connects its prompt node with weight 1 and connects
    all original nodes with learnable query-conditioned incidence weights.
    """

    def __init__(
        self,
        in_channels: int,
        num_tokens: int = 4,
        temperature: float = 1.0,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_tokens = int(max(0, num_tokens))
        self.temperature = float(max(temperature, 1e-6))
        self.prompt_features = nn.Parameter(torch.empty(self.num_tokens, self.in_channels))
        self.node_keys = nn.Parameter(torch.empty(self.num_tokens, self.in_channels))
        self.query_keys = nn.Parameter(torch.empty(self.num_tokens, self.in_channels))
        self.prompt_bias = nn.Parameter(torch.zeros(self.num_tokens))
        self.init_scale = float(init_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.num_tokens > 0:
            nn.init.normal_(self.prompt_features, std=self.init_scale)
            nn.init.normal_(self.node_keys, std=self.init_scale)
            nn.init.normal_(self.query_keys, std=self.init_scale)
        nn.init.zeros_(self.prompt_bias)

    def _graph_count(self, batch: torch.Tensor) -> int:
        if batch.numel() == 0:
            return 1
        return int(batch.max().item()) + 1

    def _num_edges(self, data: Any, hyperedge_index: torch.Tensor) -> int:
        attr_edges = _as_int(getattr(data, "num_hyperedges", None), -1)
        if attr_edges >= 0:
            return attr_edges
        if hyperedge_index.numel() == 0:
            return 0
        return int(hyperedge_index[1].max().item()) + 1

    def _query_summary(self, x_graph: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        if query_mask.numel() == x_graph.shape[0] and bool(query_mask.any()):
            return x_graph[query_mask].mean(dim=0)
        return x_graph.mean(dim=0)

    def _gates(self, x_graph: torch.Tensor, query_vec: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x_graph, p=2, dim=-1)
        q_norm = F.normalize(query_vec.view(1, -1), p=2, dim=-1)
        node_keys = F.normalize(self.node_keys, p=2, dim=-1)
        query_keys = F.normalize(self.query_keys, p=2, dim=-1)

        node_logits = x_norm.matmul(node_keys.t())
        query_logits = q_norm.matmul(query_keys.t()).view(1, -1)
        logits = (node_logits + query_logits + self.prompt_bias.view(1, -1)) / self.temperature
        return torch.sigmoid(logits)

    def forward(self, data: Any) -> Any:
        if self.num_tokens <= 0:
            return data

        x = data.x
        device = x.device
        dtype = x.dtype
        hyperedge_index = data.hyperedge_index.to(device)

        if not hasattr(data, "batch") or data.batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        else:
            batch = data.batch.to(device).long()

        query_mask = getattr(data, "query_mask", None)
        if query_mask is None:
            query_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
        else:
            query_mask = query_mask.to(device).bool().view(-1)

        graph_count = self._graph_count(batch)
        original_num_nodes = int(x.shape[0])
        original_num_edges = self._num_edges(data, hyperedge_index)
        prompt_node_start = original_num_nodes
        prompt_edge_start = original_num_edges

        prompt_x_parts = []
        prompt_batch_parts = []
        prompt_query_parts = []
        prompt_node_mask_parts = [torch.zeros(original_num_nodes, dtype=torch.bool, device=device)]
        prompt_rows = []
        prompt_cols = []
        prompt_weights = []

        for graph_id in range(graph_count):
            graph_nodes = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            if graph_nodes.numel() == 0:
                continue

            x_graph = x[graph_nodes]
            q_graph = query_mask[graph_nodes]
            query_vec = self._query_summary(x_graph, q_graph)
            gates = self._gates(x_graph, query_vec)

            prompt_node_ids = prompt_node_start + graph_id * self.num_tokens + torch.arange(
                self.num_tokens,
                dtype=torch.long,
                device=device,
            )
            prompt_edge_ids = prompt_edge_start + graph_id * self.num_tokens + torch.arange(
                self.num_tokens,
                dtype=torch.long,
                device=device,
            )

            prompt_x_parts.append(self.prompt_features)
            prompt_batch_parts.append(torch.full((self.num_tokens,), graph_id, dtype=torch.long, device=device))
            prompt_query_parts.append(torch.zeros(self.num_tokens, dtype=torch.bool, device=device))
            prompt_node_mask_parts.append(torch.ones(self.num_tokens, dtype=torch.bool, device=device))

            prompt_rows.append(prompt_node_ids)
            prompt_cols.append(prompt_edge_ids)
            prompt_weights.append(torch.ones(self.num_tokens, dtype=dtype, device=device))

            prompt_rows.append(graph_nodes.repeat_interleave(self.num_tokens))
            prompt_cols.append(prompt_edge_ids.repeat(graph_nodes.numel()))
            prompt_weights.append(gates.reshape(-1).to(dtype=dtype))

        if not prompt_x_parts:
            return data

        x_aug = torch.cat([x, torch.cat(prompt_x_parts, dim=0)], dim=0)
        batch_aug = torch.cat([batch, torch.cat(prompt_batch_parts, dim=0)], dim=0)
        query_aug = torch.cat([query_mask, torch.cat(prompt_query_parts, dim=0)], dim=0)
        prompt_node_mask = torch.cat(prompt_node_mask_parts, dim=0)

        original_weight = getattr(data, "incidence_weight", None)
        if original_weight is None:
            original_weight = torch.ones(hyperedge_index.shape[1], dtype=dtype, device=device)
        else:
            original_weight = original_weight.to(device=device, dtype=dtype).view(-1)

        prompt_hyperedge_index = torch.stack(
            [torch.cat(prompt_rows, dim=0), torch.cat(prompt_cols, dim=0)],
            dim=0,
        )
        hyperedge_aug = torch.cat([hyperedge_index, prompt_hyperedge_index], dim=1)
        incidence_weight = torch.cat([original_weight, torch.cat(prompt_weights, dim=0)], dim=0)

        prompted = copy.copy(data)
        prompted.x = x_aug
        prompted.hyperedge_index = hyperedge_aug
        prompted.edge_index = hyperedge_aug
        prompted.batch = batch_aug
        prompted.query_mask = query_aug
        prompted.prompt_node_mask = prompt_node_mask
        prompted.incidence_weight = incidence_weight
        prompted.num_nodes = int(x_aug.shape[0])
        prompted.num_hyperedges = int(original_num_edges + graph_count * self.num_tokens)
        prompted._subgraph_prepared = True
        return prompted

