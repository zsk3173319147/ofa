from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from tasker import TaskType


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


def mean_pool_by_index(values: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None) -> torch.Tensor:
    if index.numel() == 0:
        return values.mean(dim=0, keepdim=True)

    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    pooled = values.new_zeros(dim_size, values.shape[-1])
    counts = values.new_zeros(dim_size, 1)
    pooled.index_add_(0, index, values)
    counts.index_add_(0, index, torch.ones(index.shape[0], 1, dtype=values.dtype, device=values.device))
    return pooled / counts.clamp_min(1)


class SubgraphReadout(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.graph_pooling = getattr(args, "pooling", "mean")

    def reset_parameters(self) -> None:
        return

    def forward(self, node_emb: torch.Tensor, data: Any) -> torch.Tensor:
        if not hasattr(data, "batch") or data.batch is None:
            return node_emb.mean(dim=0, keepdim=True)

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
            return torch.stack(pooled, dim=0)

        return mean_pool_by_index(node_emb, batch)

    def query_pool(self, node_emb: torch.Tensor, data: Any) -> torch.Tensor:
        if not hasattr(data, "query_mask"):
            return self.forward(node_emb, data)

        query_mask = data.query_mask.to(node_emb.device).bool().view(-1)
        if query_mask.numel() != node_emb.shape[0] or not bool(query_mask.any()):
            return self.forward(node_emb, data)

        if not hasattr(data, "batch") or data.batch is None:
            return node_emb[query_mask].mean(dim=0, keepdim=True)

        batch = data.batch.to(node_emb.device).long()
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        query_batch = batch[query_mask]
        query_emb = node_emb[query_mask]

        pooled = mean_pool_by_index(query_emb, query_batch, dim_size=graph_count)
        counts = node_emb.new_zeros(graph_count, 1)
        counts.index_add_(0, query_batch, torch.ones(query_batch.shape[0], 1, dtype=node_emb.dtype, device=node_emb.device))
        missing = counts.view(-1) == 0
        if bool(missing.any()):
            pooled[missing] = self.forward(node_emb, data)[missing]
        return pooled


class SubgraphDownstreamModel(nn.Module):
    def __init__(self, encoder: nn.Module, task_type: TaskType | str, num_targets: int, args):
        super().__init__()
        self.encoder = encoder
        self.task_type = TaskType(task_type)
        self.readout = SubgraphReadout(args)
        if bool(getattr(args, "use_adapter", False)):
            if encoder.__class__.__name__ != "HGNN":
                raise ValueError("Adapter currently supports method=HGNN only.")
            encoder.enable_adapter(int(getattr(args, "adapter_hidden_dim", 16)))
        self.head = self._build_head(num_targets, args)

    def _build_head(self, num_targets: int, args) -> nn.Module:
        return nn.Linear(args.embedding_hidden, num_targets)

    def reset_parameters(self) -> None:
        if hasattr(self.encoder, "reset_parameters"):
            self.encoder.reset_parameters()
        self.readout.reset_parameters()
        if hasattr(self.head, "reset_parameters"):
            self.head.reset_parameters()

    def encode(self, data: Any, task_type: TaskType | str | None = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        reset_dynamic_encoder_state(self.encoder)
        if hasattr(self.encoder, "set_task_type"):
            self.encoder.set_task_type(task_type or self.task_type)
        return split_encoder_output(self.encoder(data))

    def forward(self, batch) -> torch.Tensor:
        h_prime = batch.h_prime
        node_emb, _ = self.encode(h_prime, batch.task_type)
        h_query = self.readout.query_pool(node_emb, h_prime)
        out = self.head(h_query)
        if batch.task_type == TaskType.EDGE_PRED:
            return out.view(-1)
        return out


class MultiTaskSubgraphModel(nn.Module):
    """Shared subgraph encoder with one lightweight head per task/dataset key."""

    def __init__(self, encoder: nn.Module, head_dims: dict[TaskType | str, int], args):
        super().__init__()
        self.encoder = encoder
        self.readout = SubgraphReadout(args)
        if bool(getattr(args, "use_adapter", False)):
            if encoder.__class__.__name__ != "HGNN":
                raise ValueError("Adapter currently supports method=HGNN only.")
            encoder.enable_adapter(int(getattr(args, "adapter_hidden_dim", 16)))
        self.heads = nn.ModuleDict(
            {str(head_key): nn.Linear(args.embedding_hidden, int(out_dim)) for head_key, out_dim in head_dims.items()}
        )

    @staticmethod
    def _head_key(batch, task_type: TaskType) -> str:
        metadata = getattr(batch, "metadata", {}) or {}
        head_key = metadata.get("head_key")
        if head_key is not None:
            return str(head_key)
        return task_type.value

    def reset_parameters(self) -> None:
        if hasattr(self.encoder, "reset_parameters"):
            self.encoder.reset_parameters()
        self.readout.reset_parameters()
        for head in self.heads.values():
            if hasattr(head, "reset_parameters"):
                head.reset_parameters()

    def encode(self, data: Any, task_type: TaskType | str) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        reset_dynamic_encoder_state(self.encoder)
        if hasattr(self.encoder, "set_task_type"):
            self.encoder.set_task_type(task_type)
        return split_encoder_output(self.encoder(data))

    def forward(self, batch) -> torch.Tensor:
        task_type = TaskType(batch.task_type)
        h_prime = batch.h_prime
        node_emb, _ = self.encode(h_prime, task_type)
        h_query = self.readout.query_pool(node_emb, h_prime)
        head_key = self._head_key(batch, task_type)
        if head_key not in self.heads:
            head_key = task_type.value
        logits = self.heads[head_key](h_query)
        if task_type == TaskType.EDGE_PRED:
            return logits.view(-1)
        return logits
