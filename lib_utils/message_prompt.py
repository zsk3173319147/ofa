from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn


class LowRankResidualPrompt(nn.Module):
    """Message-only low-rank linear residual prompt.

    For each incidence message m_i, the prompt learns

        Delta(m_i) = m_i @ A @ B

    where A and B are low-rank trainable matrices. The caller applies the
    residual strength lambda outside this module.
    """

    def __init__(self, message_dim: int, rank: int = 4, dropout: float = 0.0):
        super().__init__()
        self.message_dim = int(message_dim)
        self.rank = int(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.left = nn.Parameter(torch.empty(self.message_dim, self.rank))
        self.right = nn.Parameter(torch.empty(self.rank, self.message_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.left, a=5 ** 0.5)
        nn.init.zeros_(self.right)

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        shape = message.shape
        flat = message.reshape(shape[0], -1)
        if flat.shape[-1] != self.message_dim:
            raise ValueError(f"Prompt message dim mismatch: got {flat.shape[-1]}, expected {self.message_dim}")
        residual = self.dropout(flat) @ self.left @ self.right
        return residual.reshape(shape)


class DualFlowMessagePrompt(nn.Module):
    """Residual incidence-level HGNN message prompt.

    HGNN has two incidence message directions:

        source_to_target: node -> hyperedge
        target_to_source: hyperedge -> node

    This module keeps an independent low-rank residual prompt for each layer and
    direction:

        m' = m + lambda_{layer,direction} * Delta_{layer,direction}(m)
    """

    def __init__(
        self,
        num_layers: int,
        message_dims: Sequence[int],
        rank: int = 4,
        residual_init: float = 0.001,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = int(max(num_layers, 1))
        self.residual_init = float(residual_init)
        dims = [int(dim) for dim in message_dims]
        if len(dims) != self.num_layers:
            raise ValueError(f"message_dims length {len(dims)} does not match num_layers {self.num_layers}")

        self.residual_lambdas = nn.Parameter(torch.full((self.num_layers, 2), self.residual_init))
        self.residual_prompts = nn.ModuleList(
            nn.ModuleList(
                [
                    LowRankResidualPrompt(
                        message_dim=dim,
                        rank=rank,
                        dropout=dropout,
                    )
                    for _ in range(2)
                ]
            )
            for dim in dims
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.residual_lambdas, self.residual_init)
        for layer_prompts in self.residual_prompts:
            for prompt in layer_prompts:
                prompt.reset_parameters()

    def forward(
        self,
        message: torch.Tensor,
        node_ids: torch.Tensor | None = None,
        edge_ids: torch.Tensor | None = None,
        direction: str = "source_to_target",
        layer_id: int = 0,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if message.numel() == 0:
            return message

        direction_id = 0 if direction == "source_to_target" else 1
        layer_id = int(max(0, min(layer_id, self.num_layers - 1)))
        residual = self.residual_prompts[layer_id][direction_id](message)
        return message + self.residual_lambdas[layer_id, direction_id] * residual
