from __future__ import annotations

import copy
import os
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

from lib_dataset.edge_loaders import (
    edge_split_is_current,
    generate_edge_loaders,
    generate_ind_split_hyperedges,
    generate_split_hyperedges,
    split_positive_hyperedges,
)
from lib_dataset.hg_loaders import generate_hg_loaders, generate_split_hypergraphs
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.few_shot import apply_few_shot_edge_split, apply_few_shot_hg_split, apply_few_shot_node_split
from lib_utils.metrics import avg_result_printer_edge, edge_evaluation_printer, hg_evaluation_printer
from lib_utils.model_factory import build_edge_prediction_graph, parse_model
from lib_utils.subgraph_model import SubgraphDownstreamModel
from lib_utils.utils import fix_seed, mean_std_metrics, result_printer
from tasker import (
    GraphClsTaskAdapter,
    NodeClsSubgraphTaskAdapter,
    PropagationSubgraphBuilder,
    TaskType,
    TaskBatchDataset,
    append_graph_batch_role_features,
    build_edge_subgraph_task_batch,
    model_data_with_subgraph_schema,
)


def _prepare_hg_batch(batch, args):
    batch = algo_preprocessing(batch, args)

    if args.method in ["AllSetformer"]:
        batch.norm = torch.ones_like(batch.hyperedge_index[0])

    return batch


def _new_optimizer(model, args):
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)


class SubgraphExpAgent:
    """Downstream path: extract subgraphs, encode them with HGNN, then classify each subgraph."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.train_times = []
        self.test_dict = defaultdict(list)
        self._subgraph_builders = {}
        self._current_seed = None

    def _build_model(self, data, task_type: TaskType, num_targets: int):
        self.args.embedding_mode = True
        model_data = model_data_with_subgraph_schema(data, self.args)
        encoder = parse_model(self.args, model_data)
        model = SubgraphDownstreamModel(encoder, task_type, num_targets, self.args)
        return model.to(self.device)

    def _reset_and_prepare_model(self, model):
        model.reset_parameters()
        return _new_optimizer(model, self.args)

    def _use_best_model(self) -> bool:
        return bool(getattr(self.args, "early_stop", False)) or bool(getattr(self.args, "subgraph_use_best_model", False))

    def _save_downstream_model(self, model, task_type: TaskType | str, seed: int) -> None:
        save_path = str(getattr(self.args, "downstream_save_path", "") or "")
        if not save_path:
            return

        task_value = getattr(task_type, "value", task_type)
        task_value = str(task_value)
        if not os.path.splitext(save_path)[1]:
            save_path = os.path.join(
                save_path,
                f"{self.args.dname}_{task_value}_{self.args.method}_seed{seed}.pt",
            )
        else:
            save_path = save_path.format(
                dname=self.args.dname,
                task_type=task_value,
                method=self.args.method,
                seed=seed,
            )

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "args": vars(self.args).copy(),
                "task_type": task_value,
                "dname": self.args.dname,
                "method": self.args.method,
                "seed": int(seed),
            },
            save_path,
        )
        print(f"Saved downstream checkpoint: {save_path}")

    def _prepare_batch(self, batch):
        if not bool(getattr(batch.h_prime, "_subgraph_prepared", False)):
            batch.h_prime = _prepare_hg_batch(batch.h_prime, self.args)
            batch.h_prime._subgraph_prepared = True
        return batch.to(self.device)

    def _node_adapter(self, data, masks, split: str, shuffle: bool, builder: Optional[PropagationSubgraphBuilder] = None):
        return NodeClsSubgraphTaskAdapter(
            data,
            masks,
            self.args,
            split=split,
            batch_size=getattr(self.args, "subgraph_batch_size", 128),
            shuffle=shuffle,
            builder=builder,
        )

    def _subgraph_builder(self, data) -> PropagationSubgraphBuilder:
        key = id(data)
        builder = self._subgraph_builders.get(key)
        if builder is None:
            builder = PropagationSubgraphBuilder(data, self.args)
            self._subgraph_builders[key] = builder
        return builder

    def _train_node_epoch(self, model, adapter, optimizer):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in adapter:
            batch = self._prepare_batch(batch)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y.long())
            loss.backward()
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_thresh)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        return total_loss / max(steps, 1)

    @torch.no_grad()
    def _eval_node_split(self, model, adapter):
        model.eval()
        preds = []
        labels = []
        for batch in adapter:
            batch = self._prepare_batch(batch)
            logits = model(batch)
            preds.append(logits.argmax(dim=-1).detach().cpu())
            labels.append(batch.y.detach().cpu())

        if not preds:
            return 0.0
        pred = torch.cat(preds).numpy()
        label = torch.cat(labels).numpy()
        return accuracy_score(label, pred) * 100.0

    @torch.no_grad()
    def _eval_node(self, model, adapters):
        return [
            self._eval_node_split(model, adapters["train"]),
            self._eval_node_split(model, adapters["valid"]),
            self._eval_node_split(model, adapters["test"]),
        ]

    def node_cls_train_eval(self, data):
        metrics_dict = defaultdict(list)

        for seed in range(self.args.num_seeds):
            fix_seed(seed)
            self._current_seed = seed
            masks = data.generate_random_split(
                train_ratio=self.args.train_prop,
                val_ratio=self.args.valid_prop,
                seed=seed,
            )
            masks = apply_few_shot_node_split(data, masks, self.args, seed)
            cache_start = time.time()
            builder = self._subgraph_builder(data)
            adapters = {
                "train": self._node_adapter(data, masks, split="train", shuffle=True, builder=builder),
                "valid": self._node_adapter(data, masks, split="valid", shuffle=False, builder=builder),
                "test": self._node_adapter(data, masks, split="test", shuffle=False, builder=builder),
            }
            if bool(getattr(self.args, "subgraph_cache", True)):
                node_count = sum(int(adapter.node_ids.numel()) for adapter in adapters.values())
                batch_count = sum(len(adapter) for adapter in adapters.values())
                print(
                    f"Cached node subgraphs: {node_count} samples, "
                    f"{batch_count} batches, {time.time() - cache_start:.2f}s"
                )

            model = self._build_model(data, TaskType.NODE_CLS, data.num_classes)
            optimizer = self._reset_and_prepare_model(model)
            train_adapter = adapters["train"]

            start_time = time.time()
            best_score = -1.0
            best_model = None

            for epoch in tqdm(range(self.args.epochs)):
                loss = self._train_node_epoch(model, train_adapter, optimizer)

                if (epoch + 1) % self.args.display_step == 0:
                    result = self._eval_node(model, adapters)
                    if result[1] >= best_score:
                        best_score = result[1]
                        best_model = copy.deepcopy(model)
                    print(f"Epoch: {epoch + 1:02d}, Training loss: {loss:.4f}, Valid acc: {result[1]:.4f}")

            train_time = time.time() - start_time
            self.train_times.append(train_time)
            print(f"Training Time: {train_time:.2f}")

            eval_model = best_model if self._use_best_model() and best_model is not None else model
            result = self._eval_node(eval_model, adapters)
            if eval_model is best_model:
                print(f"Using best validation model with score: {best_score:.4f}")
            print(f"------------------------------[Seed {seed}]-----------------------------------")
            print(f"train_acc: {result[0]:.2f}, valid_acc: {result[1]:.2f}, test_acc: {result[2]:.2f}")
            print(f"------------------------------------------------------------------------------")
            self._save_downstream_model(eval_model, TaskType.NODE_CLS, seed)
            metrics_dict["acc"].append(result)

        print(f"---------------------------------[Final]--------------------------------------")
        for metric_name, values in metrics_dict.items():
            result_printer(values, metric_name)
            metrics_mean, metrics_std = mean_std_metrics(values)
            self.test_dict[metric_name].extend([metrics_mean[-1], metrics_std[-1]])
        print(f"Avg Training Time: {np.mean(self.train_times):2f}")
        print(f"------------------------------------------------------------------------------")

    def _edge_split_file(self, seed: int) -> str:
        return os.path.join(self.args.edge_save_dir, self.args.edge_split_mode, self.args.dname, f"split_{seed}.pt")

    def _ensure_edge_split(self, data, seed: int):
        split_file = self._edge_split_file(seed)
        if os.path.exists(split_file):
            data_dict = torch.load(split_file, weights_only=False)
            if edge_split_is_current(data_dict, self.args):
                return
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        if self.args.edge_split_mode == "ind":
            generate_ind_split_hyperedges(data, self.args, seed)
        elif self.args.edge_split_mode == "trand":
            generate_split_hyperedges(data, self.args, seed)
        else:
            raise NotImplementedError

    def _edge_task_batch(self, data, hyperedges, labels):
        builder = self._subgraph_builder(data)
        batch = build_edge_subgraph_task_batch(builder, hyperedges, labels)
        batch.h_prime = _prepare_hg_batch(batch.h_prime, self.args)
        return batch.to(self.device)

    def build_node_train_dataset(
        self,
        data,
        seed: int,
        source_name: Optional[str] = None,
    ) -> TaskBatchDataset:
        fix_seed(seed)
        masks = data.generate_random_split(
            train_ratio=self.args.train_prop,
            val_ratio=self.args.valid_prop,
            seed=seed,
        )
        masks = apply_few_shot_node_split(data, masks, self.args, seed)
        builder = self._subgraph_builder(data)
        adapter = self._node_adapter(data, masks, split="train", shuffle=False, builder=builder)
        batches = list(adapter)
        sample_count = sum(int(batch.y.numel()) for batch in batches)
        return TaskBatchDataset(
            name=source_name or f"{self.args.dname}:node_cls",
            task_type=TaskType.NODE_CLS,
            batches=batches,
            metadata={
                "samples": sample_count,
                "dname": self.args.dname,
            },
        )

    def _edge_train_negative_hyperedges(self, data_dict):
        if self.args.ns_method == "mixed":
            d = len(data_dict["train_sns"]) // 3
            return data_dict["train_sns"][:d] + data_dict["train_mns"][:d] + data_dict["train_cns"][:d]
        return data_dict[f"train_{self.args.ns_method}"]

    def _edge_train_task_batches(self, data, data_dict) -> list:
        pos_hyperedges = list(split_positive_hyperedges(data_dict, "train"))
        neg_hyperedges = list(self._edge_train_negative_hyperedges(data_dict))
        batch_size = max(2, int(getattr(self.args, "edge_batch_size", 512)))
        half_batch = max(1, batch_size // 2)
        num_steps = max(
            (len(pos_hyperedges) + half_batch - 1) // half_batch,
            (len(neg_hyperedges) + half_batch - 1) // half_batch,
        )

        batches = []
        builder = self._subgraph_builder(data)
        for step in range(num_steps):
            pos_part = pos_hyperedges[step * half_batch : (step + 1) * half_batch]
            neg_part = neg_hyperedges[step * half_batch : (step + 1) * half_batch]
            if not pos_part and not neg_part:
                continue
            hyperedges = pos_part + neg_part
            labels = torch.tensor([1] * len(pos_part) + [0] * len(neg_part), dtype=torch.long)
            batches.append(build_edge_subgraph_task_batch(builder, hyperedges, labels))
        return batches

    def build_edge_train_dataset(
        self,
        data,
        seed: int,
        source_name: Optional[str] = None,
    ) -> TaskBatchDataset:
        fix_seed(seed)
        self._ensure_edge_split(data, seed)
        data_dict = torch.load(self._edge_split_file(seed), weights_only=False)
        data_dict = apply_few_shot_edge_split(data_dict, self.args, seed)
        train_data = build_edge_prediction_graph(data, data_dict, self.args)
        batches = self._edge_train_task_batches(train_data, data_dict)
        sample_count = sum(int(batch.y.numel()) for batch in batches)
        return TaskBatchDataset(
            name=source_name or f"{self.args.dname}:edge_pred",
            task_type=TaskType.EDGE_PRED,
            batches=batches,
            metadata={
                "samples": sample_count,
                "dname": self.args.dname,
                "edge_split_mode": self.args.edge_split_mode,
                "ns_method": self.args.ns_method,
            },
        )

    def _train_edge_epoch(self, model, data, batch_loaders, optimizer):
        model.train()
        total_loss = 0.0
        steps = 0

        pos_loader = batch_loaders["train_pos"]
        neg_loader = batch_loaders["train_neg"]
        while True:
            pos_hyperedges, pos_labels, pos_last = pos_loader.next()
            neg_hyperedges, neg_labels, neg_last = neg_loader.next()
            hyperedges = list(pos_hyperedges) + list(neg_hyperedges)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
            batch = self._edge_task_batch(data, hyperedges, labels)

            optimizer.zero_grad()
            logits = model(batch)
            targets = batch.y.float()
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_thresh)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1
            if pos_last or neg_last:
                break

        return total_loss / max(steps, 1)

    @torch.no_grad()
    def _edge_loader_scores(self, model, data, dataloader):
        model.eval()
        preds = []
        labels = []

        while True:
            hyperedges, batch_labels, is_last = dataloader.next()
            batch = self._edge_task_batch(data, hyperedges, batch_labels)
            logits = model(batch)
            preds.append(torch.sigmoid(logits).detach().cpu())
            labels.append(batch.y.detach().cpu())
            if is_last:
                break

        return torch.cat(preds).tolist(), torch.cat(labels).tolist()

    def _eval_edge_train(self, model, data, batch_loaders):
        pred_pos, label_pos = self._edge_loader_scores(model, data, batch_loaders["train_pos"])
        pred_neg, label_neg = self._edge_loader_scores(model, data, batch_loaders["train_neg"])

        return {
            "roc_train": roc_auc_score(np.array(label_pos + label_neg), np.array(pred_pos + pred_neg)),
            "ap_train": average_precision_score(np.array(label_pos + label_neg), np.array(pred_pos + pred_neg)),
        }

    def _eval_edge_val_test(self, model, data, batch_loaders, mode: str):
        pred_pos, label_pos = self._edge_loader_scores(model, data, batch_loaders[f"{mode}_pos"])
        pred_sns, label_sns = self._edge_loader_scores(model, data, batch_loaders[f"{mode}_neg_sns"])
        pred_mns, label_mns = self._edge_loader_scores(model, data, batch_loaders[f"{mode}_neg_mns"])
        pred_cns, label_cns = self._edge_loader_scores(model, data, batch_loaders[f"{mode}_neg_cns"])

        roc_sns = roc_auc_score(np.array(label_pos + label_sns), np.array(pred_pos + pred_sns))
        ap_sns = average_precision_score(np.array(label_pos + label_sns), np.array(pred_pos + pred_sns))
        roc_mns = roc_auc_score(np.array(label_pos + label_mns), np.array(pred_pos + pred_mns))
        ap_mns = average_precision_score(np.array(label_pos + label_mns), np.array(pred_pos + pred_mns))
        roc_cns = roc_auc_score(np.array(label_pos + label_cns), np.array(pred_pos + pred_cns))
        ap_cns = average_precision_score(np.array(label_pos + label_cns), np.array(pred_pos + pred_cns))

        d = len(pred_pos) // 3
        label_mixed = label_pos + label_sns[:d] + label_mns[:d] + label_cns[:d]
        pred_mixed = pred_pos + pred_sns[:d] + pred_mns[:d] + pred_cns[:d]
        roc_mixed = roc_auc_score(np.array(label_mixed), np.array(pred_mixed))
        ap_mixed = average_precision_score(np.array(label_mixed), np.array(pred_mixed))

        return {
            "roc_sns": roc_sns,
            "ap_sns": ap_sns,
            "roc_mns": roc_mns,
            "ap_mns": ap_mns,
            "roc_cns": roc_cns,
            "ap_cns": ap_cns,
            "roc_mixed": roc_mixed,
            "ap_mixed": ap_mixed,
            "roc_average": (roc_sns + roc_mns + roc_cns + roc_mixed) / 4,
            "ap_average": (ap_sns + ap_mns + ap_cns + ap_mixed) / 4,
        }

    def _eval_edge(self, model, data, batch_loaders):
        return (
            self._eval_edge_train(model, data, batch_loaders),
            self._eval_edge_val_test(model, data, batch_loaders, "val"),
            self._eval_edge_val_test(model, data, batch_loaders, "test"),
        )

    def edge_pred_train_eval(self, data):
        metrics_dict = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

        for seed in range(self.args.num_seeds):
            fix_seed(seed)
            self._current_seed = seed
            self._ensure_edge_split(data, seed)
            data_dict = torch.load(self._edge_split_file(seed), weights_only=False)
            data_dict = apply_few_shot_edge_split(data_dict, self.args, seed)
            batch_loaders = generate_edge_loaders(data_dict, self.args)
            train_data = build_edge_prediction_graph(data, data_dict, self.args)

            model = self._build_model(train_data, TaskType.EDGE_PRED, 1)
            optimizer = self._reset_and_prepare_model(model)

            start_time = time.time()
            best_score = -1.0
            best_model = None

            for epoch in tqdm(range(self.args.epochs)):
                loss = self._train_edge_epoch(model, train_data, batch_loaders, optimizer)

                if (epoch + 1) % self.args.display_step == 0:
                    train_metrics, val_metrics, test_metrics = self._eval_edge(model, train_data, batch_loaders)
                    if val_metrics["roc_average"] >= best_score:
                        best_score = val_metrics["roc_average"]
                        best_model = copy.deepcopy(model)
                    print(f"Epoch: {epoch + 1:02d}, Training loss: {loss:.4f}")
                    edge_evaluation_printer(train_metrics, val_metrics, test_metrics)

            print(f"Training Time: {time.time() - start_time:.2f}")
            eval_model = best_model if self._use_best_model() and best_model is not None else model
            train_metrics, val_metrics, test_metrics = self._eval_edge(eval_model, train_data, batch_loaders)

            print(f"------------------------------[Seed {seed}]-----------------------------------")
            edge_evaluation_printer(train_metrics, val_metrics, test_metrics)
            print(f"------------------------------------------------------------------------------")
            self._save_downstream_model(eval_model, TaskType.EDGE_PRED, seed)

            for key, value in train_metrics.items():
                metrics_dict["train"][key].append(value)
            for key, value in val_metrics.items():
                metrics_dict["val"][key].append(value)
            for key, value in test_metrics.items():
                metrics_dict["test"][key].append(value)

        print(f"---------------------------------[Final]--------------------------------------")
        avg_result_printer_edge(metrics_dict)
        print(f"------------------------------------------------------------------------------")

    def _train_hg_epoch(self, model, adapter, optimizer):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in adapter:
            batch.h_prime = append_graph_batch_role_features(batch.h_prime, TaskType.HG_CLS, self.args)
            batch.h_prime = _prepare_hg_batch(batch.h_prime, self.args)
            batch = batch.to(self.device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y.view(-1).long())
            loss.backward()
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_thresh)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        return total_loss / max(steps, 1)

    @torch.no_grad()
    def _eval_hg_split(self, model, batch_loaders, split: str):
        model.eval()
        preds = []
        labels = []
        adapter = GraphClsTaskAdapter(batch_loaders, split=split)
        for batch in adapter:
            batch.h_prime = append_graph_batch_role_features(batch.h_prime, TaskType.HG_CLS, self.args)
            batch.h_prime = _prepare_hg_batch(batch.h_prime, self.args)
            batch = batch.to(self.device)
            logits = model(batch)
            preds.append(logits.argmax(dim=-1).detach().cpu())
            labels.append(batch.y.view(-1).detach().cpu())

        if not preds:
            return 0.0, 0.0
        pred = torch.cat(preds).numpy()
        label = torch.cat(labels).numpy()
        return accuracy_score(label, pred), f1_score(label, pred, average="macro")

    def _eval_hg(self, model, batch_loaders):
        result = defaultdict(list)
        for split in ("train", "val", "test"):
            acc, macro_f1 = self._eval_hg_split(model, batch_loaders, split)
            result["acc"].append(acc)
            result["macro_f1"].append(macro_f1)
        return result

    def hg_cls_train_eval(self, data):
        metrics_dict = defaultdict(list)

        for seed in range(self.args.num_seeds):
            fix_seed(seed)
            self._current_seed = seed
            train_set, val_set, test_set = generate_split_hypergraphs(
                data,
                self.args.train_prop,
                self.args.valid_prop,
                seed,
            )
            train_set = apply_few_shot_hg_split(train_set, self.args, seed)
            batch_loaders = generate_hg_loaders(train_set, val_set, test_set, self.args)

            model = self._build_model(data, TaskType.HG_CLS, data.num_classes)
            optimizer = self._reset_and_prepare_model(model)
            train_adapter = GraphClsTaskAdapter(batch_loaders, split="train")

            start_time = time.time()
            best_score = -1.0
            best_model = None

            for epoch in tqdm(range(self.args.epochs)):
                loss = self._train_hg_epoch(model, train_adapter, optimizer)
                if (epoch + 1) % self.args.display_step == 0:
                    result = self._eval_hg(model, batch_loaders)
                    if result["acc"][1] >= best_score:
                        best_score = result["acc"][1]
                        best_model = copy.deepcopy(model)
                    print(f"Epoch: {epoch + 1:02d}, Training loss: {loss:.4f}")
                    hg_evaluation_printer(result)

            print(f"Training Time: {time.time() - start_time:.2f}")
            eval_model = best_model if self._use_best_model() and best_model is not None else model
            result = self._eval_hg(eval_model, batch_loaders)

            print(f"------------------------------[Seed {seed}]-----------------------------------")
            hg_evaluation_printer(result)
            print(f"------------------------------------------------------------------------------")
            self._save_downstream_model(eval_model, TaskType.HG_CLS, seed)

            for metric_name, values in result.items():
                metrics_dict[metric_name].append(values)

        print(f"---------------------------------[Final]--------------------------------------")
        for metric_name, values in metrics_dict.items():
            result_printer(values, metric_name)
        print(f"------------------------------------------------------------------------------")

    def running(self, task_type, data):
        if task_type == "node_cls":
            self.node_cls_train_eval(data)
        elif task_type == "edge_pred":
            self.edge_pred_train_eval(data)
        elif task_type == "hg_cls":
            self.hg_cls_train_eval(data)
        else:
            raise NotImplementedError
