from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from lib_models.HNN import MLP
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


def mean_pool_by_index(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.numel() == 0:
        return values.mean(dim=0, keepdim=True)

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


class SubgraphDownstreamModel(nn.Module):
    def __init__(self, encoder: nn.Module, task_type: TaskType | str, num_targets: int, args):
        super().__init__()
        self.encoder = encoder
        self.task_type = TaskType(task_type)
        self.readout = SubgraphReadout(args)
        self.head = self._build_head(num_targets, args)

    def _head_type(self, args) -> str:
        head_type = getattr(args, "subgraph_head_type", "auto")
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
        node_emb, _ = self.encode(batch.h_prime)
        h_graph = self.readout(node_emb, batch.h_prime)
        out = self.head(h_graph)
        if batch.task_type == TaskType.EDGE_PRED:
            return out.view(-1)
        return out
