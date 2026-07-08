from __future__ import annotations

import copy
from typing import Any

import torch


def _target_dim(args: Any) -> int:
    return int(getattr(args, "feature_align_dim", 0) or 0)


def _align_matrix(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if target_dim <= 0:
        return x
    if x.dim() != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape={tuple(x.shape)}")

    original_device = x.device
    original_dtype = x.dtype
    x_cpu = torch.nan_to_num(x.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    node_count, feat_dim = int(x_cpu.shape[0]), int(x_cpu.shape[1])

    if feat_dim == target_dim:
        return x.to(dtype=original_dtype, device=original_device)

    if feat_dim < target_dim:
        pad = x_cpu.new_zeros((node_count, target_dim - feat_dim))
        aligned = torch.cat([x_cpu, pad], dim=-1)
        return aligned.to(dtype=original_dtype, device=original_device)

    rank = min(target_dim, node_count, feat_dim)
    if rank <= 0:
        aligned = x_cpu.new_zeros((node_count, target_dim))
        return aligned.to(dtype=original_dtype, device=original_device)

    try:
        u, s, _ = torch.pca_lowrank(x_cpu, q=rank, center=False, niter=2)
        reduced = u[:, :rank] * s[:rank]
    except Exception:
        u, s, _ = torch.linalg.svd(x_cpu, full_matrices=False)
        reduced = u[:, :rank] * s[:rank]

    if rank < target_dim:
        pad = reduced.new_zeros((node_count, target_dim - rank))
        reduced = torch.cat([reduced, pad], dim=-1)
    return reduced.to(dtype=original_dtype, device=original_device)


def _align_graph_list(graphs: list[Any], target_dim: int) -> list[Any]:
    if target_dim <= 0 or not graphs:
        return graphs

    xs = [graph.x for graph in graphs if hasattr(graph, "x") and isinstance(graph.x, torch.Tensor)]
    if not xs:
        return graphs

    counts = [int(x.shape[0]) for x in xs]
    concat = torch.cat([x.detach().cpu() for x in xs], dim=0)
    aligned_concat = _align_matrix(concat, target_dim).cpu()
    aligned_graphs = []
    offset = 0
    for graph, count in zip(graphs, counts):
        graph_copy = copy.copy(graph)
        graph_copy.x = aligned_concat[offset : offset + count].to(dtype=graph.x.dtype, device=graph.x.device)
        graph_copy.num_features = target_dim
        aligned_graphs.append(graph_copy)
        offset += count
    return aligned_graphs


def align_features(data: Any, args: Any) -> Any:
    """Project or pad feature matrices to a common multitask input dimension.

    This is intentionally a data-level adapter: it changes only the feature
    width used by pipeline=multitask, leaving single-task baseline/subgraph
    paths untouched.
    """

    target_dim = _target_dim(args)
    if target_dim <= 0 or data is None:
        return data

    # Hypergraph classification datasets in this project are lightweight
    # dataset wrappers whose ``x`` field is a list of PyG Data objects.
    if hasattr(data, "x") and isinstance(data.x, list):
        data_copy = copy.copy(data)
        data_copy.x = _align_graph_list(list(data.x), target_dim)
        data_copy.num_features = target_dim
        return data_copy

    if not hasattr(data, "x") or not isinstance(data.x, torch.Tensor):
        return data

    data_copy = copy.copy(data)
    aligned_x = _align_matrix(data.x, target_dim)
    data_copy.x = aligned_x
    data_copy.num_features = target_dim

    if hasattr(data_copy, "data") and hasattr(data_copy.data, "x"):
        data_copy.data = copy.copy(data_copy.data)
        data_copy.data.x = aligned_x
        data_copy.data.num_features = target_dim

    return data_copy
