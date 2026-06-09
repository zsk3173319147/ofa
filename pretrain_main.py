import os

import torch

from lib_dataset import _multi_datasets_, _single_datasets_
from lib_dataset.data_base import HyperDataset
from lib_dataset.edge_loaders import generate_ind_split_hyperedges, generate_split_hyperedges
from lib_dataset.preprocessing import data_processing
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.exp_agent import build_edge_prediction_graph, parse_model
from lib_utils.utils import add_self_loop_hyperedges, fix_seed, relabel_hyperedge_index
from parameter_parser import method_config, parameter_parser, pretrain_config, set_task_args
from pretrain import (
    HypeBoyHyperedgeFillingModel,
    HypeBoyHyperedgeFillingPretrainer,
    HypergraphPretrainModel,
    PretrainObjective,
    Pretrainer,
)
from tasker import (
    HyperedgeFillTaskAdapter,
    HypergraphContrastTaskAdapter,
    HypergraphDatasetContrastTaskAdapter,
    HypergraphDatasetFillTaskAdapter,
    MixedTaskLoader,
    TaskType,
)


def _parse_pretrain_tasks(tasks: str) -> list[str]:
    parsed = [task.strip().lower() for task in tasks.split(",") if task.strip()]
    aliases = {
        "hyperedge_fill": "fill",
        "ssl_hyperedge_fill": "fill",
        "hypeboy": "hypeboy_fill",
        "hypeboy_hyperedge_fill": "hypeboy_fill",
        "ssl_contrast": "contrast",
    }
    parsed = [aliases.get(task, task) for task in parsed]
    invalid = sorted(set(parsed) - {"fill", "hypeboy_fill", "contrast"})
    if invalid:
        raise ValueError(f"Unsupported pretrain task(s): {invalid}")
    return parsed


def _default_save_path(args) -> str:
    if args.pretrain_save_path:
        return args.pretrain_save_path
    return f"./pretrain_checkpoints/{args.method}_{args.dname}_pretrain.pt"


def _prepare_pretrain_graph(data, args):
    if args.task_type == "hg_cls":
        hyperedge_index, _ = relabel_hyperedge_index(data.hyperedge_index)
        data.hyperedge_index = hyperedge_index
        data.edge_index = hyperedge_index

        if args.method in ["HyperND", "TFHNN", "HyperGCN", "SheafHyperGNN"]:
            data.hyperedge_index = add_self_loop_hyperedges(data.hyperedge_index, data.num_nodes)
            data.edge_index = data.hyperedge_index

        data = algo_preprocessing(data, args)
        if args.method in ["AllSetformer", "AllDeepSets"]:
            data.norm = torch.ones_like(data.hyperedge_index[0], dtype=torch.float)
        return data

    return algo_preprocessing(data, args)


def _edge_split_file(args, seed: int) -> str:
    return os.path.join(args.edge_save_dir, args.edge_split_mode, args.dname, f"split_{seed}.pt")


def _build_edge_train_pretrain_graph(data, args):
    if args.edge_split_mode == "ind":
        generator = generate_ind_split_hyperedges
    elif args.edge_split_mode == "trand":
        generator = generate_split_hyperedges
    else:
        raise NotImplementedError

    split_file = _edge_split_file(args, args.pretrain_split_seed)
    if not os.path.exists(split_file):
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        generator(data, args, args.pretrain_split_seed)

    data_dict = torch.load(split_file, weights_only=False)
    train_data = build_edge_prediction_graph(data, data_dict, args)
    print(
        "Using edge-train pretrain graph "
        f"from split seed {args.pretrain_split_seed}: "
        f"{train_data.num_hyperedges} visible hyperedges"
    )
    return train_data


def _build_pretrain_loader(data, args):
    tasks = _parse_pretrain_tasks(args.pretrain_tasks)
    generator = torch.Generator().manual_seed(args.pretrain_seed)
    adapters = []
    graph_transform = lambda graph: _prepare_pretrain_graph(graph, args)
    is_hg_cls = args.task_type == "hg_cls"
    graph_batch_size = int(args.pretrain_hg_batch_size or args.hg_batch_size)

    if "fill" in tasks:
        if is_hg_cls:
            adapters.append(
                HypergraphDatasetFillTaskAdapter(
                    data,
                    graph_batch_size=graph_batch_size,
                    fill_batch_size=args.fill_batch_size,
                    num_negatives=args.fill_num_negatives,
                    samples_per_graph_batch=args.fill_samples_per_graph_batch,
                    generator=generator,
                    graph_transform=graph_transform,
                )
            )
        else:
            adapters.append(
                HyperedgeFillTaskAdapter(
                    data,
                    batch_size=args.fill_batch_size,
                    num_negatives=args.fill_num_negatives,
                    samples_per_epoch=args.fill_samples_per_epoch,
                    generator=generator,
                    graph_transform=graph_transform,
                )
            )

    if "contrast" in tasks:
        if is_hg_cls:
            adapters.append(
                HypergraphDatasetContrastTaskAdapter(
                    data,
                    graph_batch_size=graph_batch_size,
                    views_per_epoch=args.contrast_views_per_epoch,
                    drop_incidence_rate=args.drop_incidence_rate,
                    drop_feature_rate=args.drop_feature_rate,
                    generator=generator,
                    graph_transform=graph_transform,
                )
            )
        else:
            adapters.append(
                HypergraphContrastTaskAdapter(
                    data,
                    batch_size=args.contrast_batch_size,
                    views_per_epoch=args.contrast_views_per_epoch,
                    anchor_type=args.contrast_anchor_type,
                    drop_incidence_rate=args.drop_incidence_rate,
                    drop_feature_rate=args.drop_feature_rate,
                    generator=generator,
                    graph_transform=graph_transform,
                )
            )

    if not adapters:
        raise ValueError("At least one pretrain task is required")

    return MixedTaskLoader(adapters, strategy="round_robin")


def _save_checkpoint(model, args, history, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    checkpoint = {
        "encoder": model.encoder.state_dict(),
        "pretrain_model": model.state_dict(),
        "args": vars(args),
        "history": history,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved pretrain checkpoint to {save_path}")


def _run_hypeboy_fill_pretrain(encoder, data, args, save_path):
    if args.task_type == "hg_cls":
        raise ValueError("hypeboy_fill currently supports single-hypergraph datasets only")

    model = HypeBoyHyperedgeFillingModel(
        encoder,
        data,
        embedding_dim=args.embedding_hidden,
        projection_hidden_dim=args.hypeboy_projection_hidden,
        projection_dim=args.hypeboy_projection_dim,
        projection_dropout=args.hypeboy_projection_dropout,
        feature_mask_rate=args.hypeboy_feature_mask_rate,
        edge_drop_rate=args.hypeboy_edge_drop_rate,
        temperature=args.hypeboy_temperature,
        query_batch_size=args.hypeboy_query_batch_size,
        device=args.device,
    )
    model = model.to(args.device)
    trainer = HypeBoyHyperedgeFillingPretrainer(
        model,
        lr=args.pretrain_lr if args.pretrain_lr is not None else args.lr,
        weight_decay=args.pretrain_wd if args.pretrain_wd is not None else args.wd,
        grad_clip=args.pretrain_grad_clip,
    )
    history = trainer.fit(
        epochs=args.pretrain_epochs,
        display_step=args.pretrain_display_step,
    )
    _save_checkpoint(model, args, history, save_path)


def main():
    args = parameter_parser()
    if args.dname not in _single_datasets_ and args.dname not in _multi_datasets_:
        raise ValueError("pretrain_main.py supports single-hypergraph and hypergraph-classification datasets only")

    # Single-hypergraph datasets reuse the edge-prediction preprocessing branch:
    # it keeps the task self-supervised and avoids node-label-specific train masks.
    args.task_type = "hg_cls" if args.dname in _multi_datasets_ else "edge_pred"
    args.embedding_mode = True

    args = method_config(args)
    args = set_task_args(args)
    args = pretrain_config(args)
    args.embedding_mode = True
    fix_seed(args.pretrain_seed)

    db = HyperDataset(args)
    if args.task_type == "hg_cls":
        data = db.multi_hypergraphs
    else:
        data = data_processing(args, db)
        data._initialization_()
        if args.pretrain_graph_scope == "edge_train":
            data = _build_edge_train_pretrain_graph(data, args)
        else:
            data = algo_preprocessing(data, args)

    encoder = parse_model(args, data)
    if args.method != "TMPHN":
        encoder = encoder.to(args.device)

    tasks = _parse_pretrain_tasks(args.pretrain_tasks)
    if "hypeboy_fill" in tasks:
        if len(tasks) > 1:
            raise ValueError("hypeboy_fill is implemented as a standalone pretraining path; run it without fill/contrast")
        _run_hypeboy_fill_pretrain(encoder, data, args, _default_save_path(args))
        return

    model = HypergraphPretrainModel(encoder, embedding_dim=args.embedding_hidden)
    task_weights = {
        TaskType.SSL_HYPEREDGE_FILL: args.fill_loss_weight,
        TaskType.SSL_CONTRAST: args.contrast_loss_weight,
    }
    objective = PretrainObjective(
        task_weights=task_weights,
        contrast_temperature=args.contrast_temperature,
    )
    loader = _build_pretrain_loader(data, args)

    trainer = Pretrainer(
        model,
        objective=objective,
        lr=args.pretrain_lr if args.pretrain_lr is not None else args.lr,
        weight_decay=args.pretrain_wd if args.pretrain_wd is not None else args.wd,
        device=args.device,
        grad_clip=args.pretrain_grad_clip,
    )
    history = trainer.fit(
        loader,
        epochs=args.pretrain_epochs,
        display_step=args.pretrain_display_step,
    )

    _save_checkpoint(model, args, history, _default_save_path(args))


if __name__ == "__main__":
    main()
