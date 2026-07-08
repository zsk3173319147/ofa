from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_mean_pool

from lib_models.HNN import MLP


def split_encoder_output(output):
    if isinstance(output, (tuple, list)):
        node_emb = output[0]
        edge_emb = output[1] if len(output) > 1 else None
        return node_emb, edge_emb
    return output, None


def safe_scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None):
    if dim < 0:
        dim = src.dim() + dim
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 1

    for _ in range(src.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand_as(src)

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.full(out_shape, float("-inf"), dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(dim=dim, index=index, src=src, reduce="amax", include_self=True)
    return out


class HyperGPredictor(nn.Module):
    def __init__(self, encoder, num_targets, args):
        super().__init__()
        self.encoder = encoder
        self.pooling = args.pooling
        self.classifier = MLP(
            in_channels=args.embedding_hidden,
            hidden_channels=128,
            out_channels=num_targets,
            num_layers=2,
            dropout=0.2,
            Normalization="ln",
            InputNorm=False,
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        node_emb, _ = split_encoder_output(self.encoder(data))
        if self.pooling == "mean":
            graph_emb = global_mean_pool(node_emb, data.batch)
        elif self.pooling == "max":
            dim = -1 if isinstance(node_emb, Tensor) and node_emb.dim() == 1 else -2
            graph_emb = safe_scatter_max(node_emb, data.batch, dim=dim)
        else:
            raise ValueError(f"Unsupported graph pooling: {self.pooling}")
        return self.classifier(graph_emb)


class EdgePredictor(nn.Module):
    def __init__(self, encoder, aggregator, args):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.edge_aggr = args.edge_aggr

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.aggregator.reset_parameters()

    def encoding(self, data):
        return split_encoder_output(self.encoder(data))

    def aggregate(self, node_emb, hyperedges, mode="Train"):
        if self.edge_aggr == "group":
            size_groups = defaultdict(list)
            for hyperedge in hyperedges:
                size_groups[len(hyperedge)].append(hyperedge)

            preds = []
            for group in size_groups.values():
                he_feats = torch.stack([node_emb[hyperedge] for hyperedge in group])
                preds.append(self.aggregator(he_feats, self.edge_aggr))

            if mode == "Train":
                return torch.cat([seq.squeeze(-1) for seq in preds], dim=0)
            if mode == "Eval":
                return [pred.detach() for pred in torch.cat(preds, dim=0)]
            raise ValueError(f"Unsupported edge prediction mode: {mode}")

        preds = []
        for hyperedge in hyperedges:
            preds.append(self.aggregator(node_emb[hyperedge], self.edge_aggr))
        if mode == "Train":
            return torch.stack(preds).squeeze()
        if mode == "Eval":
            return [pred.detach() for pred in preds]
        raise ValueError(f"Unsupported edge prediction mode: {mode}")


class MeanAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classifier = MLP(
            in_channels=args.embedding_hidden,
            hidden_channels=args.e_embed_hidden,
            out_channels=1,
            num_layers=args.e_embed_layer,
            dropout=args.e_embed_dropout,
            Normalization=args.e_embed_norm,
            InputNorm=False,
        )

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def forward(self, embeddings, method="group"):
        dim = 1 if method == "group" else 0
        embedding = embeddings.mean(dim=dim).squeeze()
        embedding = torch.linalg.norm(embedding.unsqueeze(0), dim=0)
        return self.classifier(embedding)


class MaxAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classifier = MLP(
            in_channels=args.embedding_hidden,
            hidden_channels=args.e_embed_hidden,
            out_channels=1,
            num_layers=args.e_embed_layer,
            dropout=args.e_embed_dropout,
            Normalization=args.e_embed_norm,
            InputNorm=False,
        )

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def forward(self, embeddings, method):
        dim = 1 if method == "group" else 0
        embedding = torch.max(embeddings, dim=dim).values
        return self.classifier(embedding)


class MaxminAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classifier = MLP(
            in_channels=args.embedding_hidden,
            hidden_channels=args.e_embed_hidden,
            out_channels=1,
            num_layers=args.e_embed_layer,
            dropout=args.e_embed_dropout,
            Normalization=args.e_embed_norm,
            InputNorm=False,
        )

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def forward(self, embeddings, method):
        dim = 1 if method == "group" else 0
        embedding = torch.max(embeddings, dim=dim).values - torch.min(embeddings, dim=dim).values
        return self.classifier(embedding)
