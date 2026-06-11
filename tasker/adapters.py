from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Mapping

import torch

from .base import GraphQuery, TaskBatch, TaskType


class BaseTaskAdapter(Iterable[TaskBatch]):
    task_type: TaskType

    def __iter__(self) -> Iterator[TaskBatch]:
        raise NotImplementedError


class GraphClsTaskAdapter(BaseTaskAdapter):
    task_type = TaskType.HG_CLS

    def __init__(self, batch_loaders: Mapping[str, Any] | Any, split: str = "train") -> None:
        self.batch_loaders = batch_loaders
        self.split = "val" if split == "valid" else split

    def _loader(self) -> Any:
        if isinstance(self.batch_loaders, Mapping):
            return self.batch_loaders[self.split]
        return self.batch_loaders

    def __len__(self) -> int:
        loader = self._loader()
        return len(loader) if hasattr(loader, "__len__") else 0

    def __iter__(self) -> Iterator[TaskBatch]:
        for batch_id, graph_batch in enumerate(self._loader()):
            graph_count = int(graph_batch.y.numel()) if hasattr(graph_batch, "y") else None
            graph_ids = torch.arange(graph_count) if graph_count is not None else None
            yield TaskBatch(
                h_prime=graph_batch,
                query=GraphQuery(graph_ids),
                task_type=self.task_type,
                y=getattr(graph_batch, "y", None),
                split=self.split,
                metadata={"batch_id": batch_id},
            )
