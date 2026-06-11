from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import torch


class TaskType(str, Enum):
    NODE_CLS = "node_cls"
    EDGE_PRED = "edge_pred"
    HG_CLS = "hg_cls"


@dataclass
class GraphQuery:
    graph_ids: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "GraphQuery":
        if self.graph_ids is not None:
            self.graph_ids = self.graph_ids.to(device)
        return self


@dataclass
class TaskBatch:
    h_prime: Any
    query: GraphQuery
    task_type: TaskType | str
    y: Optional[torch.Tensor] = None
    split: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, TaskType):
            self.task_type = TaskType(self.task_type)

    def __iter__(self):
        yield self.h_prime
        yield self.query
        yield self.task_type
        yield self.y

    def as_tuple(
        self,
    ) -> tuple[
        Any,
        GraphQuery,
        TaskType,
        Optional[torch.Tensor],
    ]:
        return self.h_prime, self.query, self.task_type, self.y

    def to(self, device: torch.device | str) -> "TaskBatch":
        if isinstance(self.h_prime, tuple):
            self.h_prime = tuple(item.to(device) if hasattr(item, "to") else item for item in self.h_prime)
        elif isinstance(self.h_prime, list):
            self.h_prime = [item.to(device) if hasattr(item, "to") else item for item in self.h_prime]
        elif hasattr(self.h_prime, "to"):
            self.h_prime = self.h_prime.to(device)
        self.query = self.query.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        self.metadata = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.metadata.items()
        }
        return self


def normalize_split_name(split: str) -> str:
    if split == "val":
        return "valid"
    return split
