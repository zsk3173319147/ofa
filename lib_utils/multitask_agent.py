from __future__ import annotations

import copy
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib_dataset.data_base import HyperDataset
from lib_dataset.edge_loaders import generate_edge_loaders
from lib_dataset.hg_loaders import generate_hg_loaders, generate_split_hypergraphs
from lib_dataset.preprocessing import data_processing
from lib_utils.feature_alignment import align_features
from lib_utils.few_shot import apply_few_shot_edge_split, apply_few_shot_hg_split, apply_few_shot_node_split
from lib_utils.metrics import edge_evaluation_printer, hg_evaluation_printer
from lib_utils.model_factory import build_edge_prediction_graph, parse_model
from lib_utils.subgraph_agent import SubgraphExpAgent
from lib_utils.subgraph_model import MultiTaskSubgraphModel
from lib_utils.utils import fix_seed, result_printer
from tasker import (
    GraphClsTaskAdapter,
    MultiDataset,
    TaskBatchDataset,
    TaskType,
    append_graph_batch_role_features,
    model_data_with_subgraph_schema,
)


@dataclass(frozen=True)
class TaskSpec:
    dname: str
    task_type: TaskType

    @property
    def head_key(self) -> str:
        return f"{self.dname}:{self.task_type.value}"


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _float_list(text: str, count: int, default: float) -> list[float]:
    values = _split_csv(text)
    if not values:
        return [float(default)] * count
    if len(values) != count:
        raise ValueError(f"Expected {count} comma-separated values, got {len(values)}: {text}")
    return [float(value) for value in values]


def _int_list(text: str, count: int, default: int) -> list[int]:
    values = _split_csv(text)
    if not values:
        return [int(default)] * count
    if len(values) != count:
        raise ValueError(f"Expected {count} comma-separated values, got {len(values)}: {text}")
    return [int(value) for value in values]


def _with_head_key(batch, spec: TaskSpec):
    metadata = dict(getattr(batch, "metadata", {}) or {})
    metadata["head_key"] = spec.head_key
    metadata["dname"] = spec.dname
    batch.metadata = metadata
    return batch


class _HeadKeyAdapter:
    def __init__(self, adapter, spec: TaskSpec, prepare_fn=None) -> None:
        self.adapter = adapter
        self.spec = spec
        self.prepare_fn = prepare_fn

    def __len__(self) -> int:
        return len(self.adapter) if hasattr(self.adapter, "__len__") else 0

    def __iter__(self):
        for batch in self.adapter:
            if self.prepare_fn is not None:
                batch = self.prepare_fn(batch)
            yield _with_head_key(batch, self.spec)


class MultiTaskExpAgent(SubgraphExpAgent):
    """OneForAll-style mixed-task training with simple feature-dimension alignment."""

    def _task_specs(self) -> list[TaskSpec]:
        specs = []
        spec_text = str(getattr(self.args, "multitask_specs", "") or "")
        if spec_text:
            for item in _split_csv(spec_text):
                if ":" in item:
                    dname, task_name = item.split(":", 1)
                elif "=" in item:
                    dname, task_name = item.split("=", 1)
                else:
                    raise ValueError(
                        "Each multitask spec must be formatted as dname:task_type, "
                        f"got: {item}"
                    )
                specs.append(TaskSpec(dname.strip(), TaskType(task_name.strip())))
            return specs

        task_names = _split_csv(getattr(self.args, "task_names", ""))
        if not task_names:
            task_names = [str(getattr(self.args, "task_type", "node_cls"))]
        return [TaskSpec(str(self.args.dname), TaskType(task_name)) for task_name in task_names]

    def _load_spec_data(self, spec: TaskSpec):
        spec_args = copy.copy(self.args)
        spec_args.dname = spec.dname
        spec_args.task_type = spec.task_type.value

        dataset = HyperDataset(spec_args)
        if spec.task_type == TaskType.HG_CLS:
            data = dataset.multi_hypergraphs
        else:
            data = data_processing(spec_args, dataset)
            data._initialization_()

        data = align_features(data, self.args)
        print(
            f"Loaded {spec.head_key}: num_features={getattr(data, 'num_features', 'NA')}, "
            f"num_classes={getattr(data, 'num_classes', 'NA')}"
        )
        return data

    def _build_multitask_model(self, model_data_source: Any, head_dims: dict[str, int]) -> MultiTaskSubgraphModel:
        self.args.embedding_mode = True
        model_data = model_data_with_subgraph_schema(model_data_source, self.args)
        encoder = parse_model(self.args, model_data)
        model = MultiTaskSubgraphModel(encoder, head_dims, self.args)
        return model.to(self.device)

    def _node_state(self, data: Any, seed: int, spec: TaskSpec) -> dict[str, Any]:
        old_dname = self.args.dname
        self.args.dname = spec.dname
        try:
            masks = data.generate_random_split(
                train_ratio=self.args.train_prop,
                val_ratio=self.args.valid_prop,
                seed=seed,
            )
            masks = apply_few_shot_node_split(data, masks, self.args, seed)
            builder = self._subgraph_builder(data)
            adapters = {
                "train": _HeadKeyAdapter(
                    self._node_adapter(data, masks, split="train", shuffle=False, builder=builder),
                    spec,
                ),
                "valid": _HeadKeyAdapter(
                    self._node_adapter(data, masks, split="valid", shuffle=False, builder=builder),
                    spec,
                ),
                "test": _HeadKeyAdapter(
                    self._node_adapter(data, masks, split="test", shuffle=False, builder=builder),
                    spec,
                ),
            }
            batches = list(adapters["train"])
        finally:
            self.args.dname = old_dname

        sample_count = sum(int(batch.y.numel()) for batch in batches)
        train_data = TaskBatchDataset(
            name=spec.head_key,
            task_type=TaskType.NODE_CLS,
            batches=batches,
            metadata={"samples": sample_count, "dname": spec.dname, "head_key": spec.head_key},
        )
        return {
            "spec": spec,
            "train_data": train_data,
            "adapters": adapters,
            "num_targets": int(data.num_classes),
        }

    def _edge_state(self, data: Any, seed: int, spec: TaskSpec) -> dict[str, Any]:
        old_dname = self.args.dname
        self.args.dname = spec.dname
        try:
            self._ensure_edge_split(data, seed)
            data_dict = torch.load(self._edge_split_file(seed), weights_only=False)
            data_dict = apply_few_shot_edge_split(data_dict, self.args, seed)
            batch_loaders = generate_edge_loaders(data_dict, self.args)
            train_graph = build_edge_prediction_graph(data, data_dict, self.args)
            batches = self._edge_train_task_batches(train_graph, data_dict)
        finally:
            self.args.dname = old_dname

        batches = [_with_head_key(batch, spec) for batch in batches]
        sample_count = sum(int(batch.y.numel()) for batch in batches)
        train_data = TaskBatchDataset(
            name=spec.head_key,
            task_type=TaskType.EDGE_PRED,
            batches=batches,
            metadata={
                "samples": sample_count,
                "dname": spec.dname,
                "head_key": spec.head_key,
                "edge_split_mode": self.args.edge_split_mode,
                "ns_method": self.args.ns_method,
            },
        )
        return {
            "spec": spec,
            "train_data": train_data,
            "train_graph": train_graph,
            "batch_loaders": batch_loaders,
            "num_targets": 1,
        }

    def _prepare_hg_batch_for_task(self, batch, spec: TaskSpec):
        batch.h_prime = append_graph_batch_role_features(batch.h_prime, TaskType.HG_CLS, self.args)
        return batch

    def _hg_state(self, data: Any, seed: int, spec: TaskSpec) -> dict[str, Any]:
        train_set, val_set, test_set = generate_split_hypergraphs(
            data,
            self.args.train_prop,
            self.args.valid_prop,
            seed,
        )
        train_set = apply_few_shot_hg_split(train_set, self.args, seed)
        batch_loaders = generate_hg_loaders(train_set, val_set, test_set, self.args)
        train_adapter = _HeadKeyAdapter(
            GraphClsTaskAdapter(batch_loaders, split="train"),
            spec,
            prepare_fn=lambda batch: self._prepare_hg_batch_for_task(batch, spec),
        )
        batches = list(train_adapter)
        sample_count = sum(int(batch.y.numel()) for batch in batches)
        train_data = TaskBatchDataset(
            name=spec.head_key,
            task_type=TaskType.HG_CLS,
            batches=batches,
            metadata={"samples": sample_count, "dname": spec.dname, "head_key": spec.head_key},
        )
        return {
            "spec": spec,
            "train_data": train_data,
            "batch_loaders": batch_loaders,
            "num_targets": int(data.num_classes),
        }

    def _build_task_states(
        self,
        loaded_data: dict[str, Any],
        seed: int,
        specs: list[TaskSpec],
    ) -> dict[str, dict[str, Any]]:
        states = {}
        for spec in specs:
            data = loaded_data[spec.head_key]
            if spec.task_type == TaskType.NODE_CLS:
                states[spec.head_key] = self._node_state(data, seed, spec)
            elif spec.task_type == TaskType.EDGE_PRED:
                states[spec.head_key] = self._edge_state(data, seed, spec)
            elif spec.task_type == TaskType.HG_CLS:
                states[spec.head_key] = self._hg_state(data, seed, spec)
            else:
                raise NotImplementedError(spec.task_type.value)
        return states

    def _edge_task_batch(self, data, hyperedges, labels):
        batch = super()._edge_task_batch(data, hyperedges, labels)
        active_spec = getattr(self, "_active_edge_spec", None)
        if active_spec is not None:
            batch = _with_head_key(batch, active_spec)
        return batch

    def _eval_edge_for_spec(self, model, state):
        old_spec = getattr(self, "_active_edge_spec", None)
        self._active_edge_spec = state["spec"]
        try:
            return self._eval_edge(model, state["train_graph"], state["batch_loaders"])
        finally:
            self._active_edge_spec = old_spec

    @torch.no_grad()
    def _eval_hg_split_for_spec(self, model, state, split: str):
        model.eval()
        preds = []
        labels = []
        spec = state["spec"]
        adapter = _HeadKeyAdapter(
            GraphClsTaskAdapter(state["batch_loaders"], split=split),
            spec,
            prepare_fn=lambda batch: self._prepare_hg_batch_for_task(batch, spec),
        )
        for batch in adapter:
            batch = self._prepare_batch(batch)
            logits = model(batch)
            preds.append(logits.argmax(dim=-1).detach().cpu())
            labels.append(batch.y.view(-1).detach().cpu())

        if not preds:
            return 0.0, 0.0
        from sklearn.metrics import accuracy_score, f1_score

        pred = torch.cat(preds).numpy()
        label = torch.cat(labels).numpy()
        return accuracy_score(label, pred), f1_score(label, pred, average="macro")

    def _eval_hg_for_spec(self, model, state):
        result = defaultdict(list)
        for split in ("train", "val", "test"):
            acc, macro_f1 = self._eval_hg_split_for_spec(model, state, split)
            result["acc"].append(acc)
            result["macro_f1"].append(macro_f1)
        return result

    def _multitask_loss(self, model: MultiTaskSubgraphModel, batch) -> torch.Tensor:
        batch = self._prepare_batch(batch)
        logits = model(batch)
        task = TaskType(batch.task_type)
        if task == TaskType.EDGE_PRED:
            return F.binary_cross_entropy_with_logits(logits, batch.y.float())
        return F.cross_entropy(logits, batch.y.view(-1).long())

    def _train_multitask_batch(self, model, batch, optimizer) -> float:
        model.train()
        optimizer.zero_grad()
        loss = self._multitask_loss(model, batch)
        loss.backward()
        if self.args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_thresh)
        optimizer.step()
        return float(loss.item())

    def _train_multitask_epoch(self, model, train_data: MultiDataset, optimizer) -> float:
        model.train()
        order = torch.randperm(len(train_data)).tolist()
        total_loss = 0.0
        steps = 0
        for idx in order:
            batch = train_data[idx]
            total_loss += self._train_multitask_batch(model, batch, optimizer)
            steps += 1
        return total_loss / max(steps, 1)

    def _eval_multitask(self, model, states: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result = {}
        for head_key, state in states.items():
            spec = state["spec"]
            if spec.task_type == TaskType.NODE_CLS:
                metrics = self._eval_node(model, state["adapters"])
            elif spec.task_type == TaskType.EDGE_PRED:
                metrics = self._eval_edge_for_spec(model, state)
            elif spec.task_type == TaskType.HG_CLS:
                metrics = self._eval_hg_for_spec(model, state)
            else:
                raise NotImplementedError(spec.task_type.value)
            result[head_key] = {"spec": spec, "metrics": metrics}
        return result

    @staticmethod
    def _validation_score(result: dict[str, dict[str, Any]]) -> float:
        scores = []
        for item in result.values():
            spec = item["spec"]
            metrics = item["metrics"]
            if spec.task_type == TaskType.NODE_CLS:
                scores.append(float(metrics[1]) / 100.0)
            elif spec.task_type == TaskType.EDGE_PRED:
                scores.append(float(metrics[1]["roc_average"]))
            elif spec.task_type == TaskType.HG_CLS:
                scores.append(float(metrics["acc"][1]))
        return float(np.mean(scores)) if scores else 0.0

    def _print_epoch_result(self, epoch: int, loss: float, result: dict[str, dict[str, Any]]) -> None:
        parts = [f"Epoch: {epoch:02d}", f"Training loss: {loss:.4f}"]
        for head_key, item in result.items():
            spec = item["spec"]
            metrics = item["metrics"]
            if spec.task_type == TaskType.NODE_CLS:
                parts.append(f"{head_key}_valid_acc: {metrics[1]:.2f}")
            elif spec.task_type == TaskType.EDGE_PRED:
                parts.append(f"{head_key}_valid_auc: {metrics[1]['roc_average']:.4f}")
            elif spec.task_type == TaskType.HG_CLS:
                parts.append(f"{head_key}_valid_acc: {metrics['acc'][1] * 100:.2f}")
        print(", ".join(parts))

    def _print_step_result(self, step: int, loss: float, result: dict[str, dict[str, Any]]) -> None:
        parts = [f"Step: {step:04d}", f"Training loss: {loss:.4f}"]
        for head_key, item in result.items():
            spec = item["spec"]
            metrics = item["metrics"]
            if spec.task_type == TaskType.NODE_CLS:
                parts.append(f"{head_key}_valid_acc: {metrics[1]:.2f}")
            elif spec.task_type == TaskType.EDGE_PRED:
                parts.append(f"{head_key}_valid_auc: {metrics[1]['roc_average']:.4f}")
            elif spec.task_type == TaskType.HG_CLS:
                parts.append(f"{head_key}_valid_acc: {metrics['acc'][1] * 100:.2f}")
        print(", ".join(parts))

    def _print_final_result(self, result: dict[str, dict[str, Any]]) -> None:
        for head_key, item in result.items():
            spec = item["spec"]
            metrics = item["metrics"]
            if spec.task_type == TaskType.NODE_CLS:
                print(
                    f"{head_key} | "
                    f"train_acc: {metrics[0]:.2f}, valid_acc: {metrics[1]:.2f}, test_acc: {metrics[2]:.2f}"
                )
            elif spec.task_type == TaskType.EDGE_PRED:
                train_metrics, val_metrics, test_metrics = metrics
                print(f"{head_key} |")
                edge_evaluation_printer(train_metrics, val_metrics, test_metrics)
            elif spec.task_type == TaskType.HG_CLS:
                print(f"{head_key} |")
                hg_evaluation_printer(metrics)

    def running(self, task_type, data):
        specs = self._task_specs()
        loaded_data = {spec.head_key: self._load_spec_data(spec) for spec in specs}
        model_data_source = next(iter(loaded_data.values()))
        metrics = defaultdict(list)

        for seed in range(self.args.num_seeds):
            fix_seed(seed)
            self._current_seed = seed
            states = self._build_task_states(loaded_data, seed, specs)
            train_sets = [states[spec.head_key]["train_data"] for spec in specs]
            multiples = _float_list(getattr(self.args, "d_multiple", ""), len(train_sets), 1.0)
            min_ratios = _float_list(getattr(self.args, "d_min_ratio", ""), len(train_sets), 1.0)
            task_max_steps = _int_list(getattr(self.args, "task_max_steps", ""), len(train_sets), 0)
            train_data = MultiDataset(
                train_sets,
                dataset_multiple=multiples,
                min_ratio=min_ratios,
                seed=seed,
            )

            print("---------------------------------[MultiTask]--------------------------------------")
            for dataset in train_sets:
                print(
                    f"{dataset.name}: batches={len(dataset)}, "
                    f"samples={dataset.metadata.get('samples', 'NA')}"
                )
            print(f"mixed_batches_per_epoch={len(train_data)}, d_multiple={multiples}")
            if any(limit > 0 for limit in task_max_steps):
                limits_text = ", ".join(
                    f"{spec.head_key}:{limit if limit > 0 else 'inf'}"
                    for spec, limit in zip(specs, task_max_steps)
                )
                print(f"task_max_steps={limits_text}")

            head_dims = {spec.head_key: int(states[spec.head_key]["num_targets"]) for spec in specs}
            model = self._build_multitask_model(model_data_source, head_dims)
            optimizer = self._reset_and_prepare_model(model)

            best_score = -1.0
            best_model = None
            start_time = time.time()
            max_train_steps = int(getattr(self.args, "max_train_steps", 0) or 0)
            if max_train_steps > 0:
                global_step = 0
                losses = []
                eval_interval = max(1, int(self.args.display_step))
                task_step_counts = np.zeros(len(train_sets), dtype=int)
                head_to_task_index = {spec.head_key: idx for idx, spec in enumerate(specs)}

                def task_is_active(task_idx: int) -> bool:
                    limit = int(task_max_steps[task_idx])
                    return limit <= 0 or task_step_counts[task_idx] < limit

                def has_active_task() -> bool:
                    return any(task_is_active(idx) for idx in range(len(train_sets)))

                pbar = tqdm(total=max_train_steps)
                while global_step < max_train_steps and has_active_task():
                    train_data.compute_sizes()
                    order = torch.randperm(len(train_data)).tolist()
                    for idx in order:
                        batch = train_data[idx]
                        metadata = getattr(batch, "metadata", {}) or {}
                        task_idx = head_to_task_index.get(str(metadata.get("head_key", "")))
                        if task_idx is not None and not task_is_active(task_idx):
                            continue

                        loss = self._train_multitask_batch(model, batch, optimizer)
                        if task_idx is not None:
                            task_step_counts[task_idx] += 1
                        losses.append(loss)
                        global_step += 1
                        pbar.update(1)
                        should_eval = global_step % eval_interval == 0 or global_step == max_train_steps
                        if should_eval:
                            recent_loss = float(np.mean(losses[-eval_interval:]))
                            result = self._eval_multitask(model, states)
                            score = self._validation_score(result)
                            if score >= best_score:
                                best_score = score
                                best_model = copy.deepcopy(model)
                            self._print_step_result(global_step, recent_loss, result)
                        if global_step >= max_train_steps or not has_active_task():
                            break
                pbar.close()
                if any(limit > 0 for limit in task_max_steps):
                    counts_text = ", ".join(
                        f"{spec.head_key}:{int(count)}"
                        for spec, count in zip(specs, task_step_counts.tolist())
                    )
                    print(f"task_update_counts={counts_text}")
            else:
                for epoch in tqdm(range(self.args.epochs)):
                    train_data.compute_sizes()
                    loss = self._train_multitask_epoch(model, train_data, optimizer)
                    if (epoch + 1) % self.args.display_step == 0:
                        result = self._eval_multitask(model, states)
                        score = self._validation_score(result)
                        if score >= best_score:
                            best_score = score
                            best_model = copy.deepcopy(model)
                        self._print_epoch_result(epoch + 1, loss, result)

            train_time = time.time() - start_time
            eval_model = best_model if self._use_best_model() and best_model is not None else model
            result = self._eval_multitask(eval_model, states)
            print(f"Training Time: {train_time:.2f}")
            if eval_model is best_model:
                print(f"Using best multitask validation score: {best_score:.4f}")
            print(f"------------------------------[Seed {seed}]-----------------------------------")
            self._print_final_result(result)
            print("------------------------------------------------------------------------------")
            self._save_downstream_model(eval_model, "multitask", seed)

            for head_key, item in result.items():
                spec = item["spec"]
                task_metrics = item["metrics"]
                if spec.task_type == TaskType.NODE_CLS:
                    metrics[f"{head_key}:acc"].append(task_metrics)
                elif spec.task_type == TaskType.EDGE_PRED:
                    metrics[f"{head_key}:roc"].append(
                        [
                            task_metrics[0]["roc_train"],
                            task_metrics[1]["roc_average"],
                            task_metrics[2]["roc_average"],
                        ]
                    )
                elif spec.task_type == TaskType.HG_CLS:
                    metrics[f"{head_key}:acc"].append(
                        [value * 100.0 for value in task_metrics["acc"]]
                    )

        print("---------------------------------[Final]--------------------------------------")
        for metric_name, values in metrics.items():
            result_printer(values, metric_name)
        print("------------------------------------------------------------------------------")
