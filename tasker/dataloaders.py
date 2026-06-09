from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from .base import TaskBatch


class TaskDataLoader(Iterable[TaskBatch]):
    def __init__(self, adapter: Iterable[TaskBatch]) -> None:
        self.adapter = adapter

    def __iter__(self) -> Iterator[TaskBatch]:
        yield from self.adapter

    def __len__(self) -> int:
        return len(self.adapter) if hasattr(self.adapter, "__len__") else 0


class MixedTaskLoader(Iterable[TaskBatch]):
    def __init__(
        self,
        adapters: Sequence[Iterable[TaskBatch]],
        strategy: str = "round_robin",
    ) -> None:
        if strategy not in {"round_robin", "sequential"}:
            raise ValueError("strategy must be 'round_robin' or 'sequential'")
        self.adapters = list(adapters)
        self.strategy = strategy

    def __iter__(self) -> Iterator[TaskBatch]:
        if self.strategy == "sequential":
            for adapter in self.adapters:
                yield from adapter
            return

        iterators = [iter(adapter) for adapter in self.adapters]
        active = [True for _ in iterators]
        while any(active):
            for idx, iterator in enumerate(iterators):
                if not active[idx]:
                    continue
                try:
                    yield next(iterator)
                except StopIteration:
                    active[idx] = False

    def __len__(self) -> int:
        total = 0
        for adapter in self.adapters:
            if not hasattr(adapter, "__len__"):
                return 0
            total += len(adapter)
        return total
