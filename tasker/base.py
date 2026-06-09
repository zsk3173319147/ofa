from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import torch


class TaskType(str, Enum):
    NODE_CLS = "node_cls"
    EDGE_PRED = "edge_pred"
    HG_CLS = "hg_cls"
    SSL_HYPEREDGE_FILL = "ssl_hyperedge_fill"
    SSL_CONTRAST = "ssl_contrast"


@dataclass
class NodeQuery:
    node_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "NodeQuery":
        self.node_ids = self.node_ids.to(device)
        return self


@dataclass
class HyperedgeQuery:
    hyperedges: Sequence[torch.Tensor]

    def to(self, device: torch.device | str) -> "HyperedgeQuery":
        self.hyperedges = [edge.to(device) for edge in self.hyperedges]
        return self


@dataclass
class PartialHyperedgeQuery:
    contexts: Sequence[torch.Tensor]
    candidate_nodes: torch.Tensor
    hyperedge_ids: Optional[torch.Tensor] = None
    target_nodes: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "PartialHyperedgeQuery":
        self.contexts = [context.to(device) for context in self.contexts]
        self.candidate_nodes = self.candidate_nodes.to(device)
        if self.hyperedge_ids is not None:
            self.hyperedge_ids = self.hyperedge_ids.to(device)
        if self.target_nodes is not None:
            self.target_nodes = self.target_nodes.to(device)
        return self


@dataclass
class GraphQuery:
    graph_ids: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "GraphQuery":
        if self.graph_ids is not None:
            self.graph_ids = self.graph_ids.to(device)
        return self


@dataclass
class ContrastiveQuery:
    anchor_type: str
    anchor_ids: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "ContrastiveQuery":
        if self.anchor_ids is not None:
            self.anchor_ids = self.anchor_ids.to(device)
        return self


@dataclass
class TaskBatch:
    h_prime: Any
    query: NodeQuery | HyperedgeQuery | PartialHyperedgeQuery | GraphQuery | ContrastiveQuery
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
        NodeQuery | HyperedgeQuery | PartialHyperedgeQuery | GraphQuery | ContrastiveQuery,
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
