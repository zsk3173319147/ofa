import copy

import torch

from lib_dataset.edge_loaders import build_hyperedge_index_from_hyperedges, observed_hyperedges_for_embedding
from lib_dataset.preprocessing import norm_contruction
from lib_models.HNN import (
    HGNN,
    HNHN,
    PlainMLP,
    SetGNN,
    UniGCNII,
    UniGNN,
)
from lib_models.HNN.preprocessing import algo_preprocessing


def build_edge_prediction_graph(data, data_dict, args):
    edge_data = copy.deepcopy(data)
    train_hyperedges = observed_hyperedges_for_embedding(data_dict)
    hyperedge_index = build_hyperedge_index_from_hyperedges(
        train_hyperedges,
        device=edge_data.hyperedge_index.device,
    )

    edge_data.hyperedge_index = hyperedge_index
    edge_data.edge_index = hyperedge_index
    edge_data.num_hyperedges = int(hyperedge_index[1].max().item() + 1) if hyperedge_index.numel() else 0

    if hasattr(edge_data, "data"):
        edge_data.data.edge_index = hyperedge_index
        edge_data.data.hyperedge_index = hyperedge_index
        edge_data.data.num_hyperedges = torch.tensor([edge_data.num_hyperedges], device=hyperedge_index.device)

    if args.method in ["AllSetformer", "AllDeepSets"]:
        edge_data = norm_contruction(edge_data, option=args.normtype)

    edge_data = algo_preprocessing(edge_data, args)
    if hasattr(edge_data, "_initialization_"):
        edge_data._initialization_()

    return edge_data


def parse_model(args, data):
    num_targets = args.embedding_hidden if args.embedding_mode else data.num_classes

    if args.method == "AllSetformer":
        if args.LearnMask:
            model = SetGNN(data.num_features, num_targets, args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, args)
    elif args.method == "AllDeepSets":
        args.PMA = False
        args.aggregate = "add"
        if args.LearnMask:
            model = SetGNN(data.num_features, num_targets, args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, args)
    elif args.method == "HGNN":
        model = HGNN(data.num_features, num_targets, args)
    elif args.method == "HNHN":
        model = HNHN(data.num_features, num_targets, args)
    elif args.method == "UniGIN":
        model = UniGNN(data.num_features, num_targets, args)
    elif args.method == "UniGCNII":
        model = UniGCNII(data.num_features, num_targets, args)
    elif args.method == "MLP":
        model = PlainMLP(data.num_features, num_targets, args)
    else:
        raise ValueError(f"Unsupported model after cleanup: {args.method}")

    return model
