from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import torch
import torch.nn as nn

from lib_models.HNN import MLP
from lib_utils.baseline_readout import MaxAggregator, MaxminAggregator, MeanAggregator
from lib_utils.message_prompt import DualFlowMessagePrompt
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
        self.edge_aggr = getattr(args, "edge_aggr", "group")
        self.message_prompt = None
        if bool(getattr(args, "use_message_prompt", False)):
            if encoder.__class__.__name__ != "HGNN":
                raise ValueError("Message prompt currently supports method=HGNN only.")
            message_dims = [
                int(getattr(conv, "heads", 1)) * int(getattr(conv, "out_channels"))
                for conv in encoder.convs
            ]
            self.message_prompt = DualFlowMessagePrompt(
                num_layers=int(getattr(args, "All_num_layers", 1)),
                message_dims=message_dims,
                rank=int(getattr(args, "message_prompt_rank", 16)),
                residual_hidden_dim=int(getattr(args, "message_prompt_hidden_dim", 0)),
                residual_init=float(getattr(args, "message_prompt_residual_init", 0.01)),
                dropout=float(getattr(args, "message_prompt_dropout", 0.0)),
            )
            self.encoder.message_prompt = self.message_prompt
        self.head = self._build_head(num_targets, args)

    def _build_edge_aggregator(self, args) -> nn.Module:
        if args.aggr_mode == "maxmin":
            return MaxminAggregator(args)
        if args.aggr_mode == "mean":
            return MeanAggregator(args)
        if args.aggr_mode == "max":
            return MaxAggregator(args)
        raise ValueError(f"Unsupported edge aggregation mode: {args.aggr_mode}")

    def _build_head(self, num_targets: int, args) -> nn.Module:
        in_channels = args.embedding_hidden
        if self.task_type == TaskType.NODE_CLS:
            return nn.Linear(in_channels, num_targets)

        if self.task_type == TaskType.EDGE_PRED:
            return self._build_edge_aggregator(args)
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
        if self.message_prompt is not None:
            self.message_prompt.reset_parameters()
            self.encoder.message_prompt = self.message_prompt
        if hasattr(self.head, "reset_parameters"):
            self.head.reset_parameters()

    def encode(self, data: Any, task_type: TaskType | str | None = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        reset_dynamic_encoder_state(self.encoder)
        return split_encoder_output(self.encoder(data))

    def _edge_scores(self, node_emb: torch.Tensor, data: Any) -> torch.Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch.to(node_emb.device).long()
            graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        else:
            batch = torch.zeros(node_emb.shape[0], dtype=torch.long, device=node_emb.device)
            graph_count = 1

        if hasattr(data, "query_mask"):
            query_mask = data.query_mask.to(node_emb.device).bool().view(-1)
            if query_mask.numel() != node_emb.shape[0]:
                query_mask = torch.ones(node_emb.shape[0], dtype=torch.bool, device=node_emb.device)
        else:
            query_mask = torch.ones(node_emb.shape[0], dtype=torch.bool, device=node_emb.device)

        per_graph_embeddings = []
        for graph_id in range(graph_count):
            graph_mask = batch == graph_id
            values = node_emb[graph_mask & query_mask]
            if values.numel() == 0:
                values = node_emb[graph_mask]
            per_graph_embeddings.append(values)

        if self.edge_aggr == "group":
            grouped_embeddings = defaultdict(list)
            grouped_ids = defaultdict(list)
            for graph_id, embeddings in enumerate(per_graph_embeddings):
                grouped_embeddings[int(embeddings.shape[0])].append(embeddings)
                grouped_ids[int(embeddings.shape[0])].append(graph_id)

            scores = node_emb.new_zeros((graph_count,))
            for size, embeddings_list in grouped_embeddings.items():
                if size == 0:
                    continue
                he_feats = torch.stack(embeddings_list, dim=0)
                group_scores = self.head(he_feats, self.edge_aggr).view(-1)
                graph_ids = torch.tensor(grouped_ids[size], dtype=torch.long, device=node_emb.device)
                scores[graph_ids] = group_scores
            return scores

        scores = []
        for embeddings in per_graph_embeddings:
            if embeddings.numel() == 0:
                scores.append(node_emb.new_zeros(()))
            else:
                scores.append(self.head(embeddings, self.edge_aggr).view(-1)[0])
        return torch.stack(scores, dim=0)

    def forward(self, batch) -> torch.Tensor:
        node_emb, _ = self.encode(batch.h_prime, batch.task_type)
        if batch.task_type == TaskType.EDGE_PRED:
            return self._edge_scores(node_emb, batch.h_prime).view(-1)
        h_graph = self.readout(node_emb, batch.h_prime)
        out = self.head(h_graph)
        return out
