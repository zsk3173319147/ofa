from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ofa.tasker import ContrastiveQuery, PartialHyperedgeQuery, TaskBatch, TaskType
except ModuleNotFoundError:
    from tasker import ContrastiveQuery, PartialHyperedgeQuery, TaskBatch, TaskType


def split_encoder_output(output: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(output, (tuple, list)):
        node_emb = output[0]
        edge_emb = output[1] if len(output) > 1 else None
        return node_emb, edge_emb
    return output, None


def get_hyperedge_index(data: Any) -> torch.Tensor:
    if hasattr(data, "hyperedge_index"):
        return data.hyperedge_index
    if hasattr(data, "edge_index"):
        return data.edge_index
    if hasattr(data, "data"):
        return get_hyperedge_index(data.data)
    raise AttributeError("data must expose hyperedge_index, edge_index, or data.hyperedge_index")


def mean_pool_by_index(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.numel() == 0:
        return values.mean(dim=0, keepdim=True)

    dim_size = int(index.max().item()) + 1
    pooled = values.new_zeros(dim_size, values.shape[-1])
    counts = values.new_zeros(dim_size, 1)
    pooled.index_add_(0, index, values)
    counts.index_add_(0, index, torch.ones(index.shape[0], 1, dtype=values.dtype, device=values.device))
    return pooled / counts.clamp_min(1)


def select_hyperedge_embeddings(
    node_emb: torch.Tensor,
    data: Any,
    hyperedge_ids: torch.Tensor,
) -> torch.Tensor:
    hyperedge_index = get_hyperedge_index(data).to(node_emb.device)
    edge_ids = hyperedge_index[1]
    selected = []
    for edge_id in hyperedge_ids.to(edge_ids.device):
        nodes = hyperedge_index[0, edge_ids == edge_id].long()
        if nodes.numel() == 0:
            selected.append(node_emb.new_zeros(node_emb.shape[-1]))
        else:
            selected.append(node_emb[nodes].mean(dim=0))
    return torch.stack(selected, dim=0)


def select_graph_embeddings(node_emb: torch.Tensor, data: Any) -> torch.Tensor:
    if hasattr(data, "batch"):
        return mean_pool_by_index(node_emb, data.batch.to(node_emb.device))
    return node_emb.mean(dim=0, keepdim=True)


def reset_dynamic_encoder_state(encoder: nn.Module) -> None:
    for attr in ("cache", "structure", "hyperedge_attr"):
        if hasattr(encoder, attr):
            setattr(encoder, attr, None)


class HyperedgeFillHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, node_emb: torch.Tensor, query: PartialHyperedgeQuery) -> torch.Tensor:
        context_emb = []
        for context in query.contexts:
            context = context.to(node_emb.device)
            if context.numel() == 0:
                context_emb.append(node_emb.new_zeros(node_emb.shape[-1]))
            else:
                context_emb.append(node_emb[context].mean(dim=0))
        context_emb = torch.stack(context_emb, dim=0)

        candidate_nodes = query.candidate_nodes.to(node_emb.device).long()
        candidate_emb = node_emb[candidate_nodes]
        context_expanded = context_emb.unsqueeze(1).expand_as(candidate_emb)
        pair_features = torch.cat(
            [
                context_expanded,
                candidate_emb,
                context_expanded * candidate_emb,
                torch.abs(context_expanded - candidate_emb),
            ],
            dim=-1,
        )
        return self.scorer(pair_features).squeeze(-1)


class ContrastiveProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        projection_dim = projection_dim or embedding_dim
        hidden_dim = hidden_dim or embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.projector(emb)


class HypergraphPretrainModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        fill_head: Optional[HyperedgeFillHead] = None,
        contrast_head: Optional[ContrastiveProjectionHead] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.fill_head = fill_head or HyperedgeFillHead(embedding_dim)
        self.contrast_head = contrast_head or ContrastiveProjectionHead(embedding_dim)

    def reset_parameters(self) -> None:
        if hasattr(self.encoder, "reset_parameters"):
            self.encoder.reset_parameters()
        self.fill_head.reset_parameters()
        self.contrast_head.reset_parameters()

    def encode(self, data: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        reset_dynamic_encoder_state(self.encoder)
        return split_encoder_output(self.encoder(data))

    def _select_contrastive_embeddings(
        self,
        node_emb: torch.Tensor,
        edge_emb: Optional[torch.Tensor],
        data: Any,
        query: ContrastiveQuery,
    ) -> torch.Tensor:
        if query.anchor_type == "node":
            if query.anchor_ids is None:
                return node_emb
            return node_emb[query.anchor_ids.to(node_emb.device).long()]

        if query.anchor_type == "hyperedge":
            if query.anchor_ids is None:
                raise ValueError("hyperedge contrastive query requires anchor_ids")
            if edge_emb is not None and edge_emb.shape[0] > int(query.anchor_ids.max().item()):
                return edge_emb[query.anchor_ids.to(edge_emb.device).long()]
            return select_hyperedge_embeddings(node_emb, data, query.anchor_ids)

        if query.anchor_type == "graph":
            return select_graph_embeddings(node_emb, data)

        raise ValueError(f"Unsupported contrastive anchor type: {query.anchor_type}")

    def forward(self, batch: TaskBatch) -> dict[str, torch.Tensor]:
        if batch.task_type == TaskType.SSL_HYPEREDGE_FILL:
            node_emb, _ = self.encode(batch.h_prime)
            return {"logits": self.fill_head(node_emb, batch.query)}

        if batch.task_type == TaskType.SSL_CONTRAST:
            view_a, view_b = batch.h_prime
            node_a, edge_a = self.encode(view_a)
            node_b, edge_b = self.encode(view_b)
            emb_a = self._select_contrastive_embeddings(node_a, edge_a, view_a, batch.query)
            emb_b = self._select_contrastive_embeddings(node_b, edge_b, view_b, batch.query)
            return {
                "z_a": F.normalize(self.contrast_head(emb_a), dim=-1),
                "z_b": F.normalize(self.contrast_head(emb_b), dim=-1),
            }

        raise ValueError(f"Unsupported pretrain task type: {batch.task_type.value}")
