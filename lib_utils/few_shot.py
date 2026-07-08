from __future__ import annotations

import copy
import random
from typing import Any, Mapping

import torch


def _as_index(mask_or_idx: torch.Tensor) -> torch.Tensor:
    if mask_or_idx.dtype == torch.bool:
        return mask_or_idx.nonzero(as_tuple=False).view(-1).long()
    return mask_or_idx.view(-1).long()


def _sample_indices(indices: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if count <= 0 or indices.numel() == 0:
        return indices.new_empty((0,), dtype=torch.long)
    if indices.numel() <= count:
        return indices.long()
    perm = torch.randperm(indices.numel(), generator=generator)[:count]
    return indices[perm].long()


def apply_few_shot_node_split(data: Any, masks: Mapping[str, torch.Tensor], args: Any, seed: int):
    """Restrict node-classification training labels while keeping val/test unchanged."""
    k = int(getattr(args, "few_shot_k", 0) or 0)
    if k <= 0:
        return masks

    labels = (data.y if hasattr(data, "y") else data.data.y).detach().cpu().view(-1).long()
    train_ids = _as_index(masks["train"].detach().cpu())
    train_labels = labels[train_ids]
    classes = torch.unique(train_labels, sorted=True)
    if classes.numel() == 0:
        return masks

    generator = torch.Generator().manual_seed(982451653 + int(seed))
    selected = []
    scope = str(getattr(args, "few_shot_scope", "total"))

    if scope == "per_class":
        quotas = [k for _ in range(int(classes.numel()))]
    else:
        base = k // int(classes.numel())
        rem = k % int(classes.numel())
        quotas = [base + (1 if idx < rem else 0) for idx in range(int(classes.numel()))]

    for cls, quota in zip(classes.tolist(), quotas):
        cls_ids = train_ids[train_labels == int(cls)]
        selected.append(_sample_indices(cls_ids, int(quota), generator))

    selected_ids = torch.cat(selected, dim=0) if selected else train_ids.new_empty((0,), dtype=torch.long)
    if scope == "total" and selected_ids.numel() < k:
        selected_set = set(int(idx) for idx in selected_ids.tolist())
        remaining = torch.tensor(
            [int(idx) for idx in train_ids.tolist() if int(idx) not in selected_set],
            dtype=torch.long,
        )
        fill = _sample_indices(remaining, k - int(selected_ids.numel()), generator)
        selected_ids = torch.cat([selected_ids, fill], dim=0)

    if selected_ids.numel() > 0:
        selected_ids = selected_ids[torch.randperm(selected_ids.numel(), generator=generator)]

    few_masks = dict(masks)
    few_masks["train"] = selected_ids.long()
    print(f"Few-shot node train samples: {int(selected_ids.numel())} ({scope}, k={k})")
    return few_masks


def _sample_sequence(values, count: int, seed: int):
    values = list(values)
    if count <= 0 or len(values) <= count:
        return values
    rng = random.Random(seed)
    ids = rng.sample(range(len(values)), count)
    return [values[idx] for idx in ids]


def apply_few_shot_edge_split(data_dict: dict, args: Any, seed: int) -> dict:
    """Restrict edge-prediction train positives/negatives while preserving val/test.

    For edge prediction, ``few_shot_k`` means k positive hyperedges. Negative
    candidates are paired supervision rather than counted as additional shots.
    """
    k = int(getattr(args, "few_shot_k", 0) or 0)
    if k <= 0:
        return data_dict

    few = copy.deepcopy(data_dict)
    train_pos = few.get("train_only_pos", [])
    if not train_pos and few.get("ground_train"):
        train_pos = few.get("ground_train", [])
        few["ground_train"] = _sample_sequence(train_pos, k, 32452843 + int(seed))
    else:
        few["train_only_pos"] = _sample_sequence(train_pos, k, 32452843 + int(seed))

    for offset, key in enumerate(["train_sns", "train_mns", "train_cns"]):
        if key in few:
            few[key] = _sample_sequence(few[key], k, 32452843 + int(seed) + offset + 1)

    pos_count = len(few.get("train_only_pos", []) or few.get("ground_train", []))
    neg_count = len(few.get("train_sns", []))
    print(f"Few-shot edge train samples: pos={pos_count}, neg_per_type={neg_count}, k={k}")
    return few

def apply_few_shot_hg_split(train_set, args: Any, seed: int):
    """Restrict hypergraph-classification training graphs while keeping val/test unchanged."""
    k = int(getattr(args, "few_shot_k", 0) or 0)
    if k <= 0:
        return train_set

    size = len(train_set)
    if size == 0:
        return train_set

    scope = str(getattr(args, "few_shot_scope", "total"))
    generator = torch.Generator().manual_seed(49979687 + int(seed))

    labels = []
    for idx in range(size):
        y = train_set[idx].y
        labels.append(int(y.view(-1)[0].item()))
    labels = torch.tensor(labels, dtype=torch.long)

    if scope == "per_class":
        selected = []
        for cls in torch.unique(labels, sorted=True).tolist():
            cls_ids = (labels == int(cls)).nonzero(as_tuple=False).view(-1)
            selected.append(_sample_indices(cls_ids, k, generator))
        selected_ids = torch.cat(selected, dim=0) if selected else torch.empty(0, dtype=torch.long)
    else:
        all_ids = torch.arange(size, dtype=torch.long)
        selected_ids = _sample_indices(all_ids, k, generator)

    if selected_ids.numel() > 0:
        selected_ids = selected_ids[torch.randperm(selected_ids.numel(), generator=generator)]

    indices = [int(idx) for idx in selected_ids.tolist()]
    print(f"Few-shot hg train samples: {len(indices)} ({scope}, k={k})")
    return torch.utils.data.Subset(train_set, indices)

