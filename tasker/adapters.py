from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Mapping, Optional

import torch

from .base import GraphQuery, HyperedgeQuery, NodeQuery, TaskBatch, TaskType, normalize_split_name


class BaseTaskAdapter(Iterable[TaskBatch]):
    task_type: TaskType

    def __iter__(self) -> Iterator[TaskBatch]:
        raise NotImplementedError


class NodeClsTaskAdapter(BaseTaskAdapter):
    task_type = TaskType.NODE_CLS

    def __init__(
        self,
        data: Any,
        masks: Mapping[str, torch.Tensor],
        split: str = "train",
        batch_size: Optional[int] = None,
        shuffle: bool = False,
    ) -> None:
        self.data = data
        self.graph = data
        self.labels = data.y if hasattr(data, "y") else data.data.y
        self.masks = masks
        self.split = normalize_split_name(split)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _node_ids(self) -> torch.Tensor:
        mask_or_idx = self.masks[self.split]
        if mask_or_idx.dtype == torch.bool:
            return mask_or_idx.nonzero(as_tuple=False).view(-1)
        return mask_or_idx.view(-1).long()

    def __len__(self) -> int:
        node_count = self._node_ids().numel()
        if node_count == 0:
            return 0
        if self.batch_size is None:
            return 1
        return (node_count + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[TaskBatch]:
        node_ids = self._node_ids()
        if node_ids.numel() == 0:
            return
        if self.shuffle:
            node_ids = node_ids[torch.randperm(node_ids.numel(), device=node_ids.device)]

        batch_size = self.batch_size or node_ids.numel()
        for start in range(0, node_ids.numel(), batch_size):
            batch_ids = node_ids[start : start + batch_size]
            yield TaskBatch(
                h_prime=self.graph,
                query=NodeQuery(batch_ids),
                task_type=self.task_type,
                y=self.labels[batch_ids],
                split=self.split,
                metadata={"node_ids": batch_ids},
            )


class HyperedgePredTaskAdapter(BaseTaskAdapter):
    task_type = TaskType.EDGE_PRED

    def __init__(
        self,
        data: Any,
        batch_loaders: Mapping[str, Any],
        split: str = "train",
        negative: str = "mixed",
        include_positive: bool = True,
        include_negative: bool = True,
    ) -> None:
        self.data = data
        self.graph = data
        self.batch_loaders = batch_loaders
        self.split = "val" if split == "valid" else split
        self.negative = negative
        self.include_positive = include_positive
        self.include_negative = include_negative

    def _loader_keys(self) -> list[str]:
        keys = []
        if self.include_positive:
            keys.append(f"{self.split}_pos")

        if not self.include_negative:
            return keys

        if self.split == "train":
            keys.append("train_neg")
        elif self.negative == "all":
            keys.extend(
                [
                    f"{self.split}_neg_sns",
                    f"{self.split}_neg_mns",
                    f"{self.split}_neg_cns",
                ]
            )
        elif self.negative == "mixed":
            keys.extend(
                key
                for key in (
                    f"{self.split}_neg_sns",
                    f"{self.split}_neg_mns",
                    f"{self.split}_neg_cns",
                )
                if key in self.batch_loaders
            )
        else:
            keys.append(f"{self.split}_neg_{self.negative}")

        return [key for key in keys if key in self.batch_loaders]

    def __iter__(self) -> Iterator[TaskBatch]:
        for loader_key in self._loader_keys():
            loader = self.batch_loaders[loader_key]
            while True:
                hyperedges, labels, is_last = loader.next()
                yield TaskBatch(
                    h_prime=self.graph,
                    query=HyperedgeQuery(hyperedges),
                    task_type=self.task_type,
                    y=labels,
                    split=self.split,
                    metadata={"loader_key": loader_key},
                )
                if is_last:
                    break


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
