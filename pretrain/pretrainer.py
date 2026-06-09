from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

import torch

try:
    from ofa.tasker import TaskBatch
except ModuleNotFoundError:
    from tasker import TaskBatch

from .objectives import PretrainObjective


def _mean_metrics(metric_lists: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: float(sum(values) / len(values))
        for key, values in metric_lists.items()
        if values
    }


class Pretrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        objective: Optional[PretrainObjective] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str | torch.device = "cpu",
        grad_clip: Optional[float] = None,
    ) -> None:
        self.model = model
        self.objective = objective or PretrainObjective()
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_epoch(self, loader: Iterable[TaskBatch]) -> dict[str, float]:
        self.model.train()
        metrics = defaultdict(list)

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss, batch_metrics = self.objective(output, batch)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            for key, value in batch_metrics.items():
                metrics[key].append(float(value))

        return _mean_metrics(metrics)

    @torch.no_grad()
    def evaluate(self, loader: Iterable[TaskBatch]) -> dict[str, float]:
        self.model.eval()
        metrics = defaultdict(list)
        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            _, batch_metrics = self.objective(output, batch)
            for key, value in batch_metrics.items():
                metrics[key].append(float(value))
        return _mean_metrics(metrics)

    def fit(
        self,
        loader: Iterable[TaskBatch],
        epochs: int,
        eval_loader: Optional[Iterable[TaskBatch]] = None,
        display_step: int = 1,
    ) -> list[dict[str, dict[str, float]]]:
        history = []
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(loader)
            record = {"train": train_metrics}

            if eval_loader is not None:
                record["eval"] = self.evaluate(eval_loader)

            history.append(record)
            if display_step and epoch % display_step == 0:
                train_msg = ", ".join(f"{key}: {value:.4f}" for key, value in train_metrics.items())
                print(f"Pretrain Epoch {epoch:03d} | {train_msg}")
                if eval_loader is not None:
                    eval_msg = ", ".join(f"{key}: {value:.4f}" for key, value in record["eval"].items())
                    print(f"Pretrain Eval  {epoch:03d} | {eval_msg}")

        return history
