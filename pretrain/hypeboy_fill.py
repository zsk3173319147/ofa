from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import split_encoder_output


def _clone_data(data: Any) -> Any:
    if hasattr(data, "clone"):
        return data.clone()
    cloned = copy.copy(data)
    if hasattr(data, "data"):
        cloned.data = copy.copy(data.data)
    return cloned


def _get_x(data: Any) -> torch.Tensor:
    if hasattr(data, "x"):
        return data.x
    if hasattr(data, "data") and hasattr(data.data, "x"):
        return data.data.x
    raise AttributeError("data must expose x or data.x")


def _get_hyperedge_index(data: Any) -> torch.Tensor:
    if hasattr(data, "hyperedge_index"):
        return data.hyperedge_index
    if hasattr(data, "edge_index"):
        return data.edge_index
    if hasattr(data, "data"):
        return _get_hyperedge_index(data.data)
    raise AttributeError("data must expose hyperedge_index, edge_index, or data.hyperedge_index")


def _get_num_nodes(data: Any) -> int:
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        num_nodes = data.num_nodes
        return int(num_nodes.item()) if isinstance(num_nodes, torch.Tensor) else int(num_nodes)
    if hasattr(data, "n_x"):
        n_x = data.n_x
        return int(n_x.item()) if isinstance(n_x, torch.Tensor) else int(n_x)
    return int(_get_x(data).shape[0])


def _set_graph_fields(data: Any, x: torch.Tensor, hyperedge_index: torch.Tensor) -> Any:
    data.x = x
    data.hyperedge_index = hyperedge_index
    data.edge_index = hyperedge_index
    data.num_hyperedges = int(hyperedge_index[1].max().item() + 1) if hyperedge_index.numel() else 0

    if hasattr(data, "norm"):
        data.norm = torch.ones_like(hyperedge_index[0], dtype=torch.float, device=hyperedge_index.device)

    if hasattr(data, "data"):
        data.data.x = x
        data.data.hyperedge_index = hyperedge_index
        data.data.edge_index = hyperedge_index
        data.data.num_hyperedges = torch.tensor([data.num_hyperedges], device=hyperedge_index.device)
        if hasattr(data.data, "norm"):
            data.data.norm = torch.ones_like(hyperedge_index[0], dtype=torch.float, device=hyperedge_index.device)

    return data


def _hyperedges_from_index(hyperedge_index: torch.Tensor, min_size: int = 2) -> list[list[int]]:
    edge_nodes = defaultdict(list)
    for node_id, edge_id in zip(hyperedge_index[0].detach().cpu().tolist(), hyperedge_index[1].detach().cpu().tolist()):
        edge_nodes[int(edge_id)].append(int(node_id))

    hyperedges = []
    for _, nodes in sorted(edge_nodes.items()):
        nodes = sorted(set(nodes))
        if len(nodes) >= min_size:
            hyperedges.append(nodes)
    return hyperedges


def _index_from_hyperedges(hyperedges: list[list[int]], num_nodes: int, device: torch.device) -> torch.Tensor:
    rows = []
    cols = []
    for edge_id, nodes in enumerate(hyperedges):
        uniq_nodes = sorted(set(int(node) for node in nodes))
        rows.extend(uniq_nodes)
        cols.extend([edge_id] * len(uniq_nodes))

    if not rows:
        rows = list(range(num_nodes))
        cols = list(range(num_nodes))

    return torch.tensor([rows, cols], dtype=torch.long, device=device)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HypeBoyHyperedgeFillingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        data: Any,
        embedding_dim: int,
        projection_hidden_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        projection_dropout: float = 0.0,
        feature_mask_rate: float = 0.1,
        edge_drop_rate: float = 0.1,
        temperature: float = 1.0,
        query_batch_size: int = 4096,
        graph_transform=None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.data = data
        self.feature_mask_rate = feature_mask_rate
        self.edge_drop_rate = edge_drop_rate
        self.temperature = temperature
        self.query_batch_size = query_batch_size
        self.graph_transform = graph_transform
        self.device = torch.device(device)

        self.node_projection = ProjectionHead(
            embedding_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
            dropout=projection_dropout,
        )
        self.query_projection = ProjectionHead(
            embedding_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
            dropout=projection_dropout,
        )

        self.num_nodes = _get_num_nodes(data)
        self.base_hyperedge_index = _get_hyperedge_index(data).detach().cpu()
        self.hyperedges = _hyperedges_from_index(self.base_hyperedge_index)
        self.register_buffer("query_context_nodes", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("query_context_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("query_target_nodes", torch.empty(0, dtype=torch.long), persistent=False)
        self._build_queries()

    def reset_parameters(self) -> None:
        if hasattr(self.encoder, "reset_parameters"):
            self.encoder.reset_parameters()
        self.node_projection.reset_parameters()
        self.query_projection.reset_parameters()

    def _build_queries(self) -> None:
        context_nodes = []
        context_ids = []
        target_nodes = []

        for hyperedge in self.hyperedges:
            if len(hyperedge) < 2:
                continue
            for target_pos, target_node in enumerate(hyperedge):
                query_id = len(target_nodes)
                target_nodes.append(target_node)
                for node_pos, context_node in enumerate(hyperedge):
                    if node_pos == target_pos:
                        continue
                    context_nodes.append(context_node)
                    context_ids.append(query_id)

        self.query_context_nodes = torch.tensor(context_nodes, dtype=torch.long)
        self.query_context_ids = torch.tensor(context_ids, dtype=torch.long)
        self.query_target_nodes = torch.tensor(target_nodes, dtype=torch.long)

    def _augment_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mask_rate <= 0:
            return x
        keep = torch.rand(x.shape, device=x.device) >= self.feature_mask_rate
        return x * keep.to(x.dtype)

    def _augment_hyperedges(self, device: torch.device) -> torch.Tensor:
        hyperedges = self.hyperedges
        if self.edge_drop_rate > 0 and len(hyperedges) > 1:
            keep_count = max(1, int(round(len(hyperedges) * (1.0 - self.edge_drop_rate))))
            perm = torch.randperm(len(hyperedges))[:keep_count].tolist()
            kept_hyperedges = [hyperedges[idx] for idx in perm]
        else:
            kept_hyperedges = list(hyperedges)

        covered_nodes = {node for hyperedge in kept_hyperedges for node in hyperedge}
        for node_id in range(self.num_nodes):
            if node_id not in covered_nodes:
                kept_hyperedges.append([node_id])

        return _index_from_hyperedges(kept_hyperedges, self.num_nodes, device)

    def augmented_data(self) -> Any:
        x = _get_x(self.data).to(self.device)
        graph = _clone_data(self.data)
        x = self._augment_features(x)
        hyperedge_index = self._augment_hyperedges(x.device)
        graph = _set_graph_fields(graph, x, hyperedge_index)
        if self.graph_transform is not None:
            graph = self.graph_transform(graph)
        return graph

    def encode(self, data: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        for attr in ("cache", "structure", "hyperedge_attr"):
            if hasattr(self.encoder, attr):
                setattr(self.encoder, attr, None)
        return split_encoder_output(self.encoder(data))

    def query_embeddings(self, node_emb: torch.Tensor) -> torch.Tensor:
        context_nodes = self.query_context_nodes.to(node_emb.device)
        context_ids = self.query_context_ids.to(node_emb.device)
        target_nodes = self.query_target_nodes.to(node_emb.device)
        query_emb = node_emb.new_zeros(target_nodes.numel(), node_emb.shape[-1])
        query_emb.index_add_(0, context_ids, node_emb[context_nodes])
        return query_emb

    def filling_loss(self, node_emb: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        target_nodes = self.query_target_nodes.to(node_emb.device)
        if target_nodes.numel() == 0:
            raise ValueError("HypeBoy hyperedge filling requires at least one hyperedge with size >= 2")

        node_proj = F.normalize(self.node_projection(node_emb), p=2, dim=-1)
        query_proj = F.normalize(self.query_projection(self.query_embeddings(node_emb)), p=2, dim=-1)

        total_loss = node_emb.new_tensor(0.0)
        correct = 0
        total = int(target_nodes.numel())
        batch_size = max(1, int(self.query_batch_size))

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            logits = query_proj[start:end] @ node_proj.t()
            logits = logits / max(float(self.temperature), 1e-8)
            labels = target_nodes[start:end]
            total_loss = total_loss + F.cross_entropy(logits, labels, reduction="sum")
            correct += int((logits.argmax(dim=-1) == labels).sum().item())

        loss = total_loss / total
        metrics = {
            "hypeboy_fill_loss": float(loss.detach().cpu()),
            "hypeboy_fill_acc": correct / total,
        }
        return loss, metrics

    def forward(self) -> tuple[torch.Tensor, dict[str, float]]:
        graph = self.augmented_data()
        node_emb, _ = self.encode(graph)
        return self.filling_loss(node_emb)


class HypeBoyHyperedgeFillingPretrainer:
    def __init__(
        self,
        model: HypeBoyHyperedgeFillingModel,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
    ) -> None:
        self.model = model
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        loss, metrics = self.model()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return metrics

    def fit(self, epochs: int, display_step: int = 1) -> list[dict[str, dict[str, float]]]:
        history = []
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()
            history.append({"train": train_metrics})
            if display_step and epoch % display_step == 0:
                train_msg = ", ".join(f"{key}: {value:.4f}" for key, value in train_metrics.items())
                print(f"HypeBoy Fill Epoch {epoch:03d} | {train_msg}")
        return history
