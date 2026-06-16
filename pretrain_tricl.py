import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "dhgbench"))

from lib_dataset.data_base import HyperDataset
from lib_dataset.edge_loaders import (
    edge_split_is_current,
    generate_ind_split_hyperedges,
    generate_split_hyperedges,
)
from lib_dataset.preprocessing import data_processing
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.model_factory import build_edge_prediction_graph
from lib_utils.parallel_config import configure_cpu_parallelism
from lib_utils.tricl_pretrain import TriCLPretrainAgent
from lib_utils.utils import fix_seed
from parameter_parser import method_config, parameter_parser, set_task_args


configure_cpu_parallelism()


def _edge_split_path(args, seed: int) -> Path:
    return Path(args.edge_save_dir) / args.edge_split_mode / args.dname / f"split_{seed}.pt"


def _ensure_edge_split(data, args, seed: int) -> dict:
    split_path = _edge_split_path(args, seed)
    if split_path.exists():
        data_dict = torch.load(split_path, weights_only=False)
        if edge_split_is_current(data_dict, args):
            return data_dict

    split_path.parent.mkdir(parents=True, exist_ok=True)
    if args.edge_split_mode == "ind":
        generate_ind_split_hyperedges(data, args, seed)
    elif args.edge_split_mode == "trand":
        generate_split_hyperedges(data, args, seed)
    else:
        raise ValueError(f"Unsupported edge split mode: {args.edge_split_mode}")
    return torch.load(split_path, weights_only=False)


if __name__ == "__main__":
    args = parameter_parser()
    args.task_type = "node_cls"
    args.pipeline = "subgraph"
    args.embedding_mode = True
    args = method_config(args)
    args = set_task_args(args)
    args.embedding_mode = True

    fix_seed(int(args.seed))
    dataset = HyperDataset(args)
    data = data_processing(args, dataset)
    data._initialization_()

    if int(getattr(args, "tricl_edge_split_seed", -1)) >= 0:
        split_seed = int(args.tricl_edge_split_seed)
        data_dict = _ensure_edge_split(data, args, split_seed)
        data = build_edge_prediction_graph(data, data_dict, args)
        print(
            "TriCL edge split-aware pretraining: "
            f"mode={args.edge_split_mode}, split_seed={split_seed}, "
            f"visible_hyperedges={int(getattr(data, 'num_hyperedges', 0))}"
        )
    else:
        data = algo_preprocessing(data, args)

    agent = TriCLPretrainAgent(args)
    agent.run(data)
