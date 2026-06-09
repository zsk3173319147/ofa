from __future__ import annotations

from typing import Callable

from .adapters import GraphClsTaskAdapter, HyperedgePredTaskAdapter, NodeClsTaskAdapter
from .base import TaskType
from .pretrain_adapters import HyperedgeFillTaskAdapter, HypergraphContrastTaskAdapter


TaskAdapterFactory = Callable[..., object]

_TASK_ADAPTERS: dict[TaskType, TaskAdapterFactory] = {}


def register_task_adapter(task_type: TaskType | str, factory: TaskAdapterFactory) -> None:
    _TASK_ADAPTERS[TaskType(task_type)] = factory


def get_task_adapter(task_type: TaskType | str) -> TaskAdapterFactory:
    task_type = TaskType(task_type)
    if task_type not in _TASK_ADAPTERS:
        raise KeyError(f"No task adapter registered for {task_type.value}")
    return _TASK_ADAPTERS[task_type]


def build_task_adapter(task_type: TaskType | str, *args, **kwargs) -> object:
    return get_task_adapter(task_type)(*args, **kwargs)


register_task_adapter(TaskType.NODE_CLS, NodeClsTaskAdapter)
register_task_adapter(TaskType.EDGE_PRED, HyperedgePredTaskAdapter)
register_task_adapter(TaskType.HG_CLS, GraphClsTaskAdapter)
register_task_adapter(TaskType.SSL_HYPEREDGE_FILL, HyperedgeFillTaskAdapter)
register_task_adapter(TaskType.SSL_CONTRAST, HypergraphContrastTaskAdapter)
