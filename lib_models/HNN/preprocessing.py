from __future__ import annotations

import torch
import torch_scatter


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    return int(value)


def algo_preprocessing(data, args):
    if args.method == "HNHN":
        return generate_HNHN_norm(data, args)
    return data


def generate_HNHN_norm(data, args):
    edge_index = data.hyperedge_index.detach().cpu()
    device = torch.device(args.device)

    if edge_index.numel() == 0:
        num_nodes = _as_int(getattr(data, "num_nodes", None), 0)
        num_hyperedges = _as_int(getattr(data, "num_hyperedges", None), 0)
        data.D_e_alpha = torch.empty(num_hyperedges, device=device)
        data.D_v_alpha_inv = torch.empty(num_nodes, device=device)
        data.D_v_beta = torch.empty(num_nodes, device=device)
        data.D_e_beta_inv = torch.empty(num_hyperedges, device=device)
        return data

    num_nodes = _as_int(getattr(data, "num_nodes", None), int(edge_index[0].max().item()) + 1)
    num_hyperedges = _as_int(getattr(data, "num_hyperedges", None), int(edge_index[1].max().item()) + 1)
    ones = torch.ones(edge_index.shape[1], device=edge_index.device)

    alpha = args.HNHN_alpha
    beta = args.HNHN_beta

    dv = torch_scatter.scatter_add(ones, edge_index[0], dim=0, dim_size=num_nodes)
    de = torch_scatter.scatter_add(ones, edge_index[1], dim=0, dim_size=num_hyperedges)

    d_e_alpha = de ** alpha
    d_e_alpha[d_e_alpha == float("inf")] = 0
    d_v_alpha = torch_scatter.scatter_add(de[edge_index[1]], edge_index[0], dim=0, dim_size=num_nodes)

    d_v_beta = dv ** beta
    d_v_beta[d_v_beta == float("inf")] = 0
    d_e_beta = torch_scatter.scatter_add(dv[edge_index[0]], edge_index[1], dim=0, dim_size=num_hyperedges)

    d_v_alpha_inv = 1.0 / d_v_alpha
    d_v_alpha_inv[d_v_alpha_inv == float("inf")] = 0

    d_e_beta_inv = 1.0 / d_e_beta
    d_e_beta_inv[d_e_beta_inv == float("inf")] = 0

    data.D_e_alpha = d_e_alpha.float().to(device)
    data.D_v_alpha_inv = d_v_alpha_inv.float().to(device)
    data.D_v_beta = d_v_beta.float().to(device)
    data.D_e_beta_inv = d_e_beta_inv.float().to(device)
    return data
