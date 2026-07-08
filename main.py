import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'dhgbench'))
from lib_utils.parallel_config import configure_cpu_parallelism
configure_cpu_parallelism()
from lib_utils.baseline_agent import BaselineExpAgent
from lib_utils.multitask_agent import MultiTaskExpAgent
from lib_utils.subgraph_agent import SubgraphExpAgent
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_dataset.data_base import HyperDataset
from lib_dataset.preprocessing import data_processing
from parameter_parser import parameter_parser,method_config,set_task_args

if __name__ == '__main__':

    args = parameter_parser() 
    args = method_config(args)
    if args.pipeline == 'multitask':
        args.embedding_mode = False
        args.add_self_loop = False
        if args.use_bench_prop:
            args.train_prop, args.valid_prop = 0.6, 0.2
        # Multi-task validation scores mix different metrics and can prefer one task over another.
        # Use the final checkpoint by default for fair task comparison.
        args.early_stop = False
    else:
        args = set_task_args(args) 

    data = None
    if args.pipeline != 'multitask':
        data=HyperDataset(args) 

        if args.task_type == 'hg_cls':
            data = data.multi_hypergraphs 
        else:
            data = data_processing(args,data)
            data._initialization_()

            if args.task_type != 'edge_pred':
                data = algo_preprocessing(data,args)
    
    if args.pipeline == 'baseline':
        agent = BaselineExpAgent(args)
    elif args.pipeline == 'subgraph':
        agent = SubgraphExpAgent(args)
    elif args.pipeline == 'multitask':
        agent = MultiTaskExpAgent(args)
    else:
        raise ValueError(f'Unsupported pipeline: {args.pipeline}')
    agent.running(args.task_type,data)
