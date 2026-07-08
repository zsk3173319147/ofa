from __future__ import annotations

import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from lib_dataset.edge_loaders import (
    edge_split_is_current,
    generate_edge_loaders,
    generate_ind_split_hyperedges,
    generate_split_hyperedges,
)
from lib_dataset.hg_loaders import generate_hg_loaders, generate_split_hypergraphs
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.baseline_readout import EdgePredictor, HyperGPredictor, MaxAggregator, MaxminAggregator, MeanAggregator
from lib_utils.few_shot import apply_few_shot_edge_split, apply_few_shot_hg_split, apply_few_shot_node_split
from lib_utils.metrics import (
    accuracy,
    aggr_metrics,
    avg_result_printer_edge,
    edge_evaluation_printer,
    evaluate,
    evaluate_edge,
    evaluate_hypegraph,
    hg_evaluation_printer,
)
from lib_utils.model_factory import build_edge_prediction_graph, parse_model
from lib_utils.utils import fix_seed, mean_std_metrics, result_printer


def _trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _new_optimizer(model, args):
    if args.method == "UniGCNII" and hasattr(model, "reg_params"):
        return torch.optim.Adam(
            [
                {"params": model.reg_params, "weight_decay": 0.01},
                {"params": model.non_reg_params, "weight_decay": 5e-4},
            ],
            lr=0.01,
        )
    return torch.optim.Adam(_trainable_parameters(model), lr=args.lr, weight_decay=args.wd)


def _prepare_hg_batch(batch, args):
    batch = algo_preprocessing(batch, args)
    if args.method == "AllSetformer":
        batch.norm = torch.ones_like(batch.hyperedge_index[0])
    return batch.to(args.device)


class BaselineExpAgent:
    """Original downstream baseline: train on the original hypergraph task readout."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.train_times = []
        self.test_dict = defaultdict(list)

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
            raise ValueError(f"Unsupported edge split mode: {self.args.edge_split_mode}")

    def _edge_aggregator(self):
        if self.args.aggr_mode == "maxmin":
            return MaxminAggregator(self.args)
        if self.args.aggr_mode == "mean":
            return MeanAggregator(self.args)
        if self.args.aggr_mode == "max":
            return MaxAggregator(self.args)
        raise ValueError(f"Unsupported edge aggregation mode: {self.args.aggr_mode}")

    def _train_node_epoch(self, model, data, masks, optimizer):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data)
        out = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(out[masks["train"]], data.y[masks["train"]])
        loss.backward()
        if self.args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_thresh)
        optimizer.step()
        return float(loss.item())

    def node_cls_train_eval(self, data):
        metrics_dict = defaultdict(list)

        for seed in range(self.args.num_seeds):
            fix_seed(seed)
            masks = data.generate_random_split(
                train_ratio=self.args.train_prop,
                val_ratio=self.args.valid_prop,
                seed=seed,
            )
            masks = apply_few_shot_node_split(data, masks, self.args, seed)

            self.args.embedding_mode = False
            model = parse_model(self.args, data).to(self.device)
            model.reset_parameters()
            optimizer = _new_optimizer(model, self.args)

            start_time = time.time()
            best_score = -1.0
            best_model = None

            for epoch in trange(self.args.epochs):
                self._train_node_epoch(model, data, masks, optimizer)

                if (epoch + 1) % self.args.display_step == 0:
                    result = evaluate(model, data, masks)
                    print(
                        f"Epoch: {epoch + 1:02d}, "
                        f"Train Acc: {100 * result[0]:.2f}%, "
                        f"Valid Acc: {100 * result[1]:.2f}%, "
                        f"Test  Acc: {100 * result[2]:.2f}%"
                    )
                    if result[1] >= best_score:
                        best_score = result[1]
                        best_model = copy.deepcopy(model)

            train_time = time.time() - start_time
            self.train_times.append(train_time)
            print(f"Training Time: {train_time:.2f}")

            eval_model = best_model if self.args.early_stop and best_model is not None else model
            if eval_model is best_model:
                print(f"Using best validation model with acc: {100 * best_score:.2f}%")
            result = {"acc": accuracy(eval_model(data)[0], data.y, masks)}

            print(f"------------------------------[Seed {seed}]-----------------------------------")
            print(f"train_acc: {result['acc'][0]:.2f}, valid_acc: {result['acc'][1]:.2f}, test_acc: {result['acc'][2]:.2f} ")
            print(f"------------------------------------------------------------------------------")
            for metric_name, values in result.items():
                metrics_dict[metric_name].append(values)

        print(f"---------------------------------[Final]--------------------------------------")
        for metric_name, values in metrics_dict.items():
            result_printer(values, metric_name)
            metrics_mean, metrics_std = mean_std_metrics(values)
            self.test_dict[metric_name].extend([metrics_mean[-1], metrics_std[-1]])
        print(f"Avg Training Time: {np.mean(self.train_times):2f}")
        print(f"------------------------------------------------------------------------------")

    def _train_edge_epoch(self, model, data, batch_loaders, optimizer):
        model.train()
        total_loss = 0.0
        train_pos_loader = batch_loaders["train_pos"]
        train_neg_loader = batch_loaders["train_neg"]

        while True:
            optimizer.zero_grad()
            node_emb, _ = model.encoding(data)

            pos_hyperedges, pos_labels, is_last = train_pos_loader.next()
            neg_hyperedges, neg_labels, is_last = train_neg_loader.next()
            pos_labels = pos_labels.to(self.device)
            neg_labels = neg_labels.to(self.device)

            pos_preds = model.aggregate(node_emb, pos_hyperedges, mode="Train")
            neg_preds = model.aggregate(node_emb, neg_hyperedges, mode="Train")

            loss = F.binary_cross_entropy_with_logits(pos_preds, pos_labels) + F.binary_cross_entropy_with_logits(neg_preds, neg_labels)
            loss.backward()
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_thresh)
            optimizer.step()
            total_loss += float(loss.item())

            if is_last:
                break

        return total_loss

    def edge_pred_train_eval(self, data):
        metrics_dict = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

        for seed in range(self.args.num_seeds):
            fix_seed(seed)
            self._ensure_edge_split(data, seed)
            data_dict = torch.load(self._edge_split_file(seed), weights_only=False)
            data_dict = apply_few_shot_edge_split(data_dict, self.args, seed)
            batch_loaders = generate_edge_loaders(data_dict, self.args)
            train_data = build_edge_prediction_graph(data, data_dict, self.args).to(self.device)

            self.args.embedding_mode = True
            encoder = parse_model(self.args, train_data)
            model = EdgePredictor(encoder, self._edge_aggregator(), self.args).to(self.device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(_trainable_parameters(model), lr=self.args.lr, weight_decay=self.args.wd)

            start_time = time.time()
            best_score = -1.0
            best_model = None

            for epoch in tqdm(range(self.args.epochs)):
                loss = self._train_edge_epoch(model, train_data, batch_loaders, optimizer)
                if (epoch + 1) % self.args.display_step == 0:
                    print(f"Epoch: {epoch + 1:02d}, Training loss: {loss:.4f}")
                    train_metrics, val_metrics, test_metrics = evaluate_edge(model, train_data, batch_loaders)
                    if val_metrics["roc_average"] >= best_score:
                        best_score = val_metrics["roc_average"]
                        best_model = copy.deepcopy(model)
                    edge_evaluation_printer(train_metrics, val_metrics, test_metrics)

            print(f"Training Time: {time.time() - start_time:.2f}")
            eval_model = best_model if self.args.early_stop and best_model is not None else model
            result = {"train": None, "val": None, "test": None}

            print(f"------------------------------[Seed {seed}]-----------------------------------")
            train_metrics, val_metrics, test_metrics = evaluate_edge(eval_model, train_data, batch_loaders)
            edge_evaluation_printer(train_metrics, val_metrics, test_metrics)
            print(f"------------------------------------------------------------------------------")
            result["train"], result["val"], result["test"] = train_metrics, val_metrics, test_metrics
            metrics_dict = aggr_metrics(metrics_dict, result)

        print(f"---------------------------------[Final]--------------------------------------")
        avg_result_printer_edge(metrics_dict)
        print(f"------------------------------------------------------------------------------")

    def _train_hg_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            batch = _prepare_hg_batch(batch, self.args)
            optimizer.zero_grad()
            out = F.log_softmax(model(batch), dim=1)
            loss = criterion(out, batch.y)
            loss.backward()
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_thresh)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        return total_loss / max(steps, 1)

    def hg_cls_train_eval(self, data):
        metrics_dict = defaultdict(list)

        for seed in range(self.args.num_seeds):
            fix_seed(seed)
            train_set, val_set, test_set = generate_split_hypergraphs(data, self.args.train_prop, self.args.valid_prop, seed)
            train_set = apply_few_shot_hg_split(train_set, self.args, seed)
            batch_loaders = generate_hg_loaders(train_set, val_set, test_set, self.args)

            self.args.embedding_mode = True
            encoder = parse_model(self.args, data)
            model = HyperGPredictor(encoder, data.num_classes, self.args).to(self.device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(_trainable_parameters(model), lr=self.args.lr, weight_decay=self.args.wd)
            criterion = torch.nn.NLLLoss()

            start_time = time.time()
            best_score = -1.0
            best_model = None

            for epoch in tqdm(range(self.args.epochs)):
                loss = self._train_hg_epoch(model, batch_loaders["train"], optimizer, criterion)
                if (epoch + 1) % self.args.display_step == 0:
                    result = evaluate_hypegraph(model, batch_loaders, self.args)
                    if result["acc"][1] >= best_score:
                        best_score = result["acc"][1]
                        best_model = copy.deepcopy(model)
                    print(f"Epoch: {epoch + 1:02d}, Training loss: {loss:.4f}")
                    hg_evaluation_printer(result)

            print(f"Training Time: {time.time() - start_time:.2f}")
            eval_model = best_model if self.args.early_stop and best_model is not None else model
            print(f"------------------------------[Seed {seed}]-----------------------------------")
            result = evaluate_hypegraph(eval_model, batch_loaders, self.args)
            hg_evaluation_printer(result)
            print(f"------------------------------------------------------------------------------")
            for metric_name, values in result.items():
                metrics_dict[metric_name].append(values)

        print(f"---------------------------------[Final]--------------------------------------")
        for metric_name, values in metrics_dict.items():
            result_printer(metrics_dict[metric_name], metric_name)
        print(f"------------------------------------------------------------------------------")

    def running(self, task_type, data):
        if task_type == "node_cls":
            self.node_cls_train_eval(data)
        elif task_type == "edge_pred":
            self.edge_pred_train_eval(data)
        elif task_type == "hg_cls":
            self.hg_cls_train_eval(data)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
