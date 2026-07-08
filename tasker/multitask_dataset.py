from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import TaskBatch, TaskType


@dataclass
class TaskBatchDataset:
    name: str
    task_type: TaskType | str
    batches: Sequence[TaskBatch]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, TaskType):
            self.task_type = TaskType(self.task_type)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[int(index)]


class MultiDataset:
    """OneForAll-style wrapper for mixing multiple task datasets."""

    def __init__(
        self,
        datas,
        data_val_index=None,
        dataset_multiple=1,
        window_size=5,
        patience=3,
        min_ratio=1,
        mode=None,
        seed=None,
    ):
        self.datas = list(datas)
        self.sizes = np.array([len(d) for d in self.datas])
        self.performance_record = []
        self.patience = patience
        self.data_val_index = data_val_index
        if self.data_val_index is None:
            self.data_val_index = [[i] for i in range(len(self.datas))]
        if isinstance(self.patience, int):
            self.patience = np.zeros(len(self.sizes)) + self.patience
        self.inpatience = np.zeros(len(self.patience))
        self.window_size = window_size
        if isinstance(self.window_size, int):
            self.window_size = np.zeros(len(self.sizes)) + self.window_size
        self.dataset_multiple = dataset_multiple
        if not isinstance(self.dataset_multiple, list):
            self.dataset_multiple = np.zeros(len(self.sizes), dtype=float) + self.dataset_multiple
        self.min_ratio = min_ratio
        if isinstance(self.min_ratio, (int, float)):
            self.min_ratio = np.zeros(len(self.sizes), dtype=float) + self.min_ratio
        self.mode = mode
        if mode is not None:
            self.mode = np.array([1 if m == "max" else -1 for m in self.mode])
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self.compute_sizes()

    def compute_sizes(self):
        self.aug_sizes = (self.sizes * np.array(self.dataset_multiple)).astype(int)
        self.size_seg = np.cumsum(self.aug_sizes)
        self.ind2dataset = np.arange(len(self.datas)).repeat(self.aug_sizes)
        repeated_sizes = self.sizes.repeat(self.aug_sizes)
        self.sample_ind = (
            (self.rng.rand(len(self.ind2dataset)) * repeated_sizes).astype(int)
            if len(repeated_sizes) > 0
            else np.array([], dtype=int)
        )
        self.data_start_index = np.r_[0, self.size_seg[:-1]]

    def __len__(self):
        return int(np.sum(self.aug_sizes))

    def __getitem__(self, index):
        dataset_ind = self.ind2dataset[index]
        dataset = self.datas[dataset_ind]
        return dataset[self.sample_ind[index]]

    def update(self, metric):
        metric = np.array(metric)
        p_records = np.array(self.performance_record)
        for i in range(len(self.datas)):
            if len(p_records) < self.window_size[i] or len(self.data_val_index[i]) == 0:
                continue

            vals = p_records[-int(self.window_size[i]):, self.data_val_index[i]]
            if self.mode is None:
                mode = np.ones(len(vals[0]), dtype=float)
            else:
                mode = self.mode[self.data_val_index[i]]
            mean = vals.mean()
            metric_vals = metric[self.data_val_index[i]]
            mean_improvement = (((metric_vals - mean) / mean) * mode).sum()
            if mean_improvement > 0:
                self.inpatience[i] = 0
            else:
                self.inpatience[i] += 1
            if self.inpatience[i] > self.patience[i]:
                self.dataset_multiple[i] = max(
                    self.min_ratio[i],
                    self.dataset_multiple[i] / 2,
                )
        self.compute_sizes()
        self.performance_record.append(metric)
