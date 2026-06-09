from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F

try:
    from ofa.tasker import TaskBatch, TaskType
except ModuleNotFoundError:
    from tasker import TaskBatch, TaskType


def hyperedge_fill_loss(output: Mapping[str, torch.Tensor], batch: TaskBatch) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["logits"]
    labels = batch.y.to(logits.device).long()
    loss = F.cross_entropy(logits, labels)
    pred = logits.argmax(dim=-1)
    acc = (pred == labels).float().mean().item()
    return loss, {"fill_acc": acc}


def contrastive_loss(
    output: Mapping[str, torch.Tensor],
    temperature: float = 0.2,
    symmetric: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    z_a = output["z_a"]
    z_b = output["z_b"]
    logits = z_a @ z_b.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.t(), labels))
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, {"contrast_acc": acc}


class PretrainObjective:
    def __init__(
        self,
        task_weights: Optional[Mapping[TaskType | str, float]] = None,
        contrast_temperature: float = 0.2,
        symmetric_contrast: bool = True,
    ) -> None:
        self.task_weights = {
            TaskType(task_type): weight for task_type, weight in (task_weights or {}).items()
        }
        self.contrast_temperature = contrast_temperature
        self.symmetric_contrast = symmetric_contrast

    def __call__(self, output: Mapping[str, torch.Tensor], batch: TaskBatch) -> tuple[torch.Tensor, dict[str, float]]:
        if batch.task_type == TaskType.SSL_HYPEREDGE_FILL:
            loss, metrics = hyperedge_fill_loss(output, batch)
        elif batch.task_type == TaskType.SSL_CONTRAST:
            loss, metrics = contrastive_loss(
                output,
                temperature=self.contrast_temperature,
                symmetric=self.symmetric_contrast,
            )
        else:
            raise ValueError(f"Unsupported pretrain task type: {batch.task_type.value}")

        weight = self.task_weights.get(batch.task_type, 1.0)
        weighted_loss = loss * weight
        metrics = dict(metrics)
        metrics["loss"] = float(loss.detach().cpu())
        metrics["weighted_loss"] = float(weighted_loss.detach().cpu())
        return weighted_loss, metrics
