from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TaskConditionedMessagePrompt(nn.Module):
    """Task-conditioned low-rank residual for HGNN incidence messages.

    For an incidence message m, the prompt computes:

        z_m = m A
        z_c = MLP([task_onehot, direction_onehot])
        Delta(m) = phi([z_m, z_c]) B
        m' = m + Delta(m)

    The final projection B is zero-initialized, so the module starts as an
    exact no-op and learns a task-specific message residual during downstream
    adaptation.
    """

    TASK_TO_ID = {
        "node_cls": 0,
        "edge_pred": 1,
        "hg_cls": 2,
    }
    DIRECTION_TO_ID = {
        "source_to_target": 0,
        "target_to_source": 1,
    }

    def __init__(self, channels: int, rank: int = 4, condition_mode: str = "task_direction") -> None:
        super().__init__()
        self.channels = int(channels)
        self.rank = int(max(1, rank))
        if condition_mode not in {"task_direction", "direction", "none"}:
            raise ValueError(f"Unsupported message prompt condition mode: {condition_mode}")
        self.condition_mode = condition_mode

        self.down = nn.Parameter(torch.empty(self.channels, self.rank))
        self.up = nn.Parameter(torch.empty(self.rank, self.channels))
        self.condition_mlp = nn.Sequential(
            nn.Linear(5, self.rank),
            nn.Tanh(),
            nn.Linear(self.rank, self.rank),
        )
        self.mix = nn.Sequential(
            nn.Linear(self.rank * 2, self.rank),
            nn.Tanh(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.down, std=0.02)
        nn.init.zeros_(self.up)
        for module in self.condition_mlp.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.mix.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    @staticmethod
    def _normalize_task(task_type: Any) -> str:
        if task_type is None:
            return "node_cls"
        value = getattr(task_type, "value", task_type)
        value = str(value)
        if value.startswith("TaskType."):
            value = value.split(".", 1)[1].lower()
        return value

    def _condition(self, message: torch.Tensor, task_type: Any, direction: str | None) -> torch.Tensor:
        if self.condition_mode == "none":
            return message.new_zeros(1, self.rank)

        task_key = self._normalize_task(task_type)
        task_id = self.TASK_TO_ID.get(task_key, 0)
        direction_id = self.DIRECTION_TO_ID.get(str(direction), 0)

        condition = message.new_zeros(5)
        if self.condition_mode == "task_direction":
            condition[task_id] = 1.0
        condition[3 + direction_id] = 1.0
        return self.condition_mlp(condition.view(1, -1))

    def forward(
        self,
        message: torch.Tensor,
        task_type: Any = None,
        direction: str | None = None,
    ) -> torch.Tensor:
        original_shape = message.shape
        flat = message.reshape(-1, self.channels)

        message_code = flat.matmul(self.down)
        condition_code = self._condition(message, task_type, direction).expand(flat.shape[0], -1)
        prompt_code = self.mix(torch.cat([message_code, condition_code], dim=-1))
        delta = prompt_code.matmul(self.up).reshape(original_shape)
        return message + delta
