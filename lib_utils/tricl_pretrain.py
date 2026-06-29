from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.model_factory import parse_model
from tasker import PropagationSubgraphBuilder, TaskType, model_data_with_subgraph_schema
from tasker.subgraph_adapters import subgraph_role_dim


TRICL_TAU_NODE = 0.5
TRICL_TAU_GROUP = 0.5
TRICL_TAU_MEMBERSHIP = 1.0
TRICL_WEIGHT_GROUP = 1.0
TRICL_WEIGHT_MEMBERSHIP = 1.0
TRICL_MAX_CONTRAST_ITEMS = 2048
TRICL_MAX_MEMBERSHIPS = 8192
TRICL_RECIPE = {
    "seed_mode": "mixed",
    "tau_node": TRICL_TAU_NODE,
    "tau_group": TRICL_TAU_GROUP,
    "tau_membership": TRICL_TAU_MEMBERSHIP,
    "weight_group": TRICL_WEIGHT_GROUP,
    "weight_membership": TRICL_WEIGHT_MEMBERSHIP,
    "proj_dim": "embedding_hidden",
    "proj_dropout": 0.0,
    "max_contrast_items": TRICL_MAX_CONTRAST_ITEMS,
    "max_memberships": TRICL_MAX_MEMBERSHIPS,
}


def _prepare_hg_batch(batch: Data, args) -> Data:
    batch = algo_preprocessing(batch, args)
    if args.method in ["AllSetformer"]:
        batch.norm = torch.ones_like(batch.hyperedge_index[0])
    return batch


def _projector(dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _symmetric_info_nce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float,
    max_items: int = 0,
) -> torch.Tensor:
    if z1.numel() == 0 or z2.numel() == 0:
        return z1.new_zeros(())

    count = min(z1.shape[0], z2.shape[0])
    if count <= 1:
        return z1.new_zeros(())

    z1 = z1[:count]
    z2 = z2[:count]
    if max_items > 0 and count > max_items:
        perm = torch.randperm(count, device=z1.device)[:max_items]
        z1 = z1[perm]
        z2 = z2[perm]
        count = int(max_items)

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(count, device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _sample_different(total: int, avoid: torch.Tensor) -> torch.Tensor:
    if total <= 1:
        return avoid.new_zeros(avoid.shape)
    sampled = torch.randint(total - 1, avoid.shape, device=avoid.device)
    return sampled + (sampled >= avoid).long()


def _membership_loss(
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
    incidence: torch.Tensor,
    temperature: float,
    max_memberships: int = 0,
) -> torch.Tensor:
    if incidence.numel() == 0 or node_emb.numel() == 0 or edge_emb.numel() == 0:
        return node_emb.new_zeros(())

    rows = incidence[0].to(node_emb.device).long()
    cols = incidence[1].to(node_emb.device).long()
    valid = (rows >= 0) & (rows < node_emb.shape[0]) & (cols >= 0) & (cols < edge_emb.shape[0])
    rows = rows[valid]
    cols = cols[valid]
    if rows.numel() == 0:
        return node_emb.new_zeros(())

    if max_memberships > 0 and rows.numel() > max_memberships:
        perm = torch.randperm(rows.numel(), device=node_emb.device)[:max_memberships]
        rows = rows[perm]
        cols = cols[perm]

    node_emb = F.normalize(node_emb, dim=-1)
    edge_emb = F.normalize(edge_emb, dim=-1)

    pos = (node_emb[rows] * edge_emb[cols]).sum(dim=-1) / temperature
    neg_edges = _sample_different(edge_emb.shape[0], cols)
    neg_e = (node_emb[rows] * edge_emb[neg_edges]).sum(dim=-1) / temperature
    loss_node_anchor = F.cross_entropy(torch.stack([pos, neg_e], dim=-1), torch.zeros_like(rows))

    neg_nodes = _sample_different(node_emb.shape[0], rows)
    neg_v = (node_emb[neg_nodes] * edge_emb[cols]).sum(dim=-1) / temperature
    loss_edge_anchor = F.cross_entropy(torch.stack([pos, neg_v], dim=-1), torch.zeros_like(rows))
    return 0.5 * (loss_node_anchor + loss_edge_anchor)


def _drop_feature_view(batch: Data, drop_rate: float, role_dim: int) -> Data:
    view = batch.clone()
    if drop_rate <= 0:
        return view
    x = view.x.clone()
    base_dim = x.shape[-1] - max(role_dim, 0)
    if base_dim <= 0:
        view.x = x
        return view
    keep = torch.rand(base_dim, device=x.device) >= drop_rate
    x[:, :base_dim] = x[:, :base_dim] * keep.to(dtype=x.dtype).view(1, -1)
    view.x = x
    return view


def _drop_incidence_view(batch: Data, drop_rate: float) -> Data:
    view = batch.clone()
    edge_index = view.hyperedge_index
    if drop_rate <= 0 or edge_index.numel() == 0:
        return view

    device = edge_index.device
    rows, cols = edge_index[0], edge_index[1]
    keep = torch.rand(rows.shape[0], device=device) >= drop_rate

    num_edges = int(getattr(view, "num_hyperedges", 0) or 0)
    if num_edges <= 0:
        num_edges = int(cols.max().item()) + 1

    # Keep every local hyperedge represented so hyperedge IDs stay aligned
    # across two augmented views.
    for edge_id in range(num_edges):
        edge_members = (cols == edge_id).nonzero(as_tuple=False).view(-1)
        if edge_members.numel() == 0:
            continue
        if not bool(keep[edge_members].any()):
            keep[edge_members[torch.randint(edge_members.numel(), (1,), device=device)[0]]] = True

    kept_edge_index = edge_index[:, keep]
    if kept_edge_index.numel() == 0:
        kept_edge_index = edge_index[:, :1]
    view.hyperedge_index = kept_edge_index
    view.edge_index = kept_edge_index
    view.num_hyperedges = num_edges
    return view


def make_tricl_view(batch: Data, args) -> Data:
    role_dim = subgraph_role_dim(args)
    view = _drop_feature_view(batch, float(args.tricl_drop_feature_rate), role_dim)
    view = _drop_incidence_view(view, float(args.tricl_drop_incidence_rate))
    return _prepare_hg_batch(view, args)


@dataclass
class TriCLBatchSeeds:
    seeds: list[list[int]]
    labels: torch.Tensor


def _sample_seeds(builder: PropagationSubgraphBuilder, args) -> TriCLBatchSeeds:
    batch_size = int(args.tricl_batch_size)
    num_nodes = int(builder.x.shape[0])
    edge_nodes = [nodes for nodes in builder.edge_nodes if len(nodes) > 0]

    seeds: list[list[int]] = []
    node_count = batch_size // 2
    node_ids = torch.randint(num_nodes, (node_count,)).tolist()
    seeds.extend([[int(node_id)] for node_id in node_ids])

    edge_count = batch_size - len(seeds)
    if edge_nodes:
        for _ in range(edge_count):
            seeds.append([int(node) for node in random.choice(edge_nodes)])
    else:
        node_ids = torch.randint(num_nodes, (edge_count,)).tolist()
        seeds.extend([[int(node_id)] for node_id in node_ids])

    labels = torch.zeros(len(seeds), dtype=torch.long)
    return TriCLBatchSeeds(seeds=seeds, labels=labels)


class TriCLPretrainModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.node_projection = _projector(hidden_dim, hidden_dim, proj_dim, dropout)
        self.edge_projection = _projector(hidden_dim, hidden_dim, proj_dim, dropout)

    def reset_parameters(self):
        if hasattr(self.encoder, "reset_parameters"):
            self.encoder.reset_parameters()
        for module in [self.node_projection, self.edge_projection]:
            for child in module.modules():
                if hasattr(child, "reset_parameters"):
                    child.reset_parameters()

    def forward(self, batch: Data) -> tuple[torch.Tensor, torch.Tensor]:
        node_emb, edge_emb = self.encoder(batch)
        if edge_emb is None:
            raise ValueError("TriCL pretraining requires an encoder that returns hyperedge embeddings.")
        return self.node_projection(node_emb), self.edge_projection(edge_emb)


def tricl_loss(
    model: TriCLPretrainModel,
    batch: Data,
    args,
) -> tuple[torch.Tensor, dict[str, float]]:
    view1 = make_tricl_view(batch, args)
    view2 = make_tricl_view(batch, args)

    z1, y1 = model(view1)
    z2, y2 = model(view2)
    incidence = batch.hyperedge_index.to(z1.device)

    loss_node = _symmetric_info_nce(
        z1,
        z2,
        TRICL_TAU_NODE,
        max_items=TRICL_MAX_CONTRAST_ITEMS,
    )
    loss_group = _symmetric_info_nce(
        y1,
        y2,
        TRICL_TAU_GROUP,
        max_items=TRICL_MAX_CONTRAST_ITEMS,
    )
    loss_mem12 = _membership_loss(
        z1,
        y2,
        incidence,
        TRICL_TAU_MEMBERSHIP,
        max_memberships=TRICL_MAX_MEMBERSHIPS,
    )
    loss_mem21 = _membership_loss(
        z2,
        y1,
        incidence,
        TRICL_TAU_MEMBERSHIP,
        max_memberships=TRICL_MAX_MEMBERSHIPS,
    )
    loss_membership = 0.5 * (loss_mem12 + loss_mem21)

    loss = loss_node + TRICL_WEIGHT_GROUP * loss_group + TRICL_WEIGHT_MEMBERSHIP * loss_membership
    metrics = {
        "loss_node": float(loss_node.detach().cpu()),
        "loss_group": float(loss_group.detach().cpu()),
        "loss_membership": float(loss_membership.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, metrics


class TriCLPretrainAgent:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

    def _build_model(self, data) -> TriCLPretrainModel:
        self.args.embedding_mode = True
        model_data = model_data_with_subgraph_schema(data, self.args)
        encoder = parse_model(self.args, model_data)
        model = TriCLPretrainModel(
            encoder,
            hidden_dim=int(self.args.embedding_hidden),
            proj_dim=int(self.args.embedding_hidden),
            dropout=0.0,
        )
        return model.to(self.device)

    def _checkpoint_path(self) -> str:
        if self.args.tricl_save_path:
            return self.args.tricl_save_path
        split_seed = int(getattr(self.args, "tricl_edge_split_seed", -1))
        split_tag = ""
        if split_seed >= 0:
            split_tag = f"_{self.args.edge_split_mode}_split{split_seed}"
        filename = (
            f"{self.args.dname}_{self.args.method}_tricl_"
            f"hop{self.args.subgraph_context_hops}_he{self.args.subgraph_max_hyperedges}"
            f"{split_tag}.pt"
        )
        return os.path.join(self.args.pretrain_save_dir, filename)

    def run(self, data) -> str:
        builder = PropagationSubgraphBuilder(data, self.args)
        model = self._build_model(data)
        model.reset_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

        best_loss = float("inf")
        best_state = None
        start_time = time.time()

        for epoch in range(1, int(self.args.epochs) + 1):
            model.train()
            epoch_metrics = {"loss": 0.0, "loss_node": 0.0, "loss_group": 0.0, "loss_membership": 0.0}
            steps = int(self.args.tricl_steps_per_epoch)
            progress = tqdm(range(steps), desc=f"TriCL epoch {epoch:03d}", leave=False)
            for _ in progress:
                seeds = _sample_seeds(builder, self.args)
                batch = builder.build_batch(seeds.seeds, seeds.labels, TaskType.EDGE_PRED, exclude_exact_seed_edge=False)
                batch = _prepare_hg_batch(batch, self.args).to(self.device)

                optimizer.zero_grad(set_to_none=True)
                loss, metrics = tricl_loss(model, batch, self.args)
                loss.backward()
                if self.args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_thresh)
                optimizer.step()

                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                progress.set_postfix(loss=f"{metrics['loss']:.4f}")

            for key in epoch_metrics:
                epoch_metrics[key] /= max(steps, 1)

            if epoch_metrics["loss"] < best_loss:
                best_loss = epoch_metrics["loss"]
                best_state = {
                    "encoder": model.encoder.state_dict(),
                    "args": vars(self.args).copy(),
                    "tricl_recipe": TRICL_RECIPE,
                    "epoch": epoch,
                    "loss": best_loss,
                }

            if epoch % int(self.args.display_step) == 0 or epoch == 1 or epoch == int(self.args.epochs):
                print(
                    f"Epoch: {epoch:03d}, "
                    f"loss: {epoch_metrics['loss']:.4f}, "
                    f"node: {epoch_metrics['loss_node']:.4f}, "
                    f"group: {epoch_metrics['loss_group']:.4f}, "
                    f"membership: {epoch_metrics['loss_membership']:.4f}"
                )

        checkpoint = self._checkpoint_path()
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        torch.save(
            best_state
            or {"encoder": model.encoder.state_dict(), "args": vars(self.args).copy(), "tricl_recipe": TRICL_RECIPE},
            checkpoint,
        )
        print(f"Training Time: {time.time() - start_time:.2f}")
        print(f"Saved TriCL encoder checkpoint: {checkpoint}")
        return checkpoint
