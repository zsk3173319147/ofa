import os
import copy
import numpy as np
#from parameter_parser import parameter_parser
import torch
from collections import defaultdict
from lib_utils.utils import fix_seed,result_printer,mean_std_metrics
from lib_utils.train_agent import Trainer
from lib_utils.eval_agent import Evaluator
from lib_models.HNN import HCHA,HyperGCN,HNHN,SetGNN,UniGNN,UniGCNII,LEGCN,HyperND,EquivSetGNN,\
                            PlainUnigencoder,HJRL,SheafHyperGNN,EHNN,TMPHN,PhenomNN,PhenomNNS,DPHGNN,TFHNN,PlainMLP,HyperGT,CEGCN,CEGAT

from lib_dataset.data_perturbation import perturbation
from lib_dataset.edge_loaders import (
    build_hyperedge_index_from_hyperedges,
    generate_edge_loaders,
    generate_split_hyperedges,
    generate_ind_split_hyperedges,
    split_positive_hyperedges,
)
from lib_dataset.hg_loaders import generate_split_hypergraphs,generate_hg_loaders
from lib_utils.aggregator import EdgePredictor,MeanAggregator,MaxminAggregator,MaxAggregator,HyperGPredictor,NodePredictor
from lib_utils.metrics import (
    aggr_metrics,
    avg_result_printer_edge,
    edge_evaluation_printer,
    evaluate_edge_fill_zero_shot,
)
from lib_dataset.preprocessing import norm_contruction
from lib_models.HNN.preprocessing import algo_preprocessing
try:
    from ofa.pretrain import HypergraphPretrainModel
except ModuleNotFoundError:
    from pretrain import HypergraphPretrainModel

class ExpAgent:
    
    def __init__(self,args,**kwargs):
        """
        Overall pipline for different kinds of models
        """
        self.args = args
        self.device=args.device
        self.trainer=Trainer(args)
        self.evaluator=Evaluator(args)
        self.train_times = []

    def edge_pred_train_eval(self,data):
        
        metrics_dict = {'train':defaultdict(list),'val':defaultdict(list),'test':defaultdict(list)}
        
        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            dir_path = os.path.join(self.args.edge_save_dir, self.args.edge_split_mode, self.args.dname, "")
            
            if self.args.edge_split_mode == 'ind':

                file_path = dir_path+f"split_{seed}.pt"
                if not os.path.exists(file_path):
                    os.makedirs(dir_path, exist_ok=True)
                    generate_ind_split_hyperedges(data,self.args,seed)

            elif self.args.edge_split_mode == 'trand':
                
                file_path = dir_path+f"split_{seed}.pt"
                if not os.path.exists(file_path):
                    os.makedirs(dir_path, exist_ok=True)
                    generate_split_hyperedges(data,self.args,seed)

            else:

                raise NotImplementedError
                
            data_dict = torch.load(file_path, weights_only=False)
            batch_loaders = generate_edge_loaders(data_dict,self.args)
            train_data = build_edge_prediction_graph(data, data_dict, self.args)

            if self.args.downstream_mode == 'zero_shot':
                model = self.build_zero_shot_pretrain_model(train_data)
                train_metrics, val_metrics, test_metrics = evaluate_edge_fill_zero_shot(model,train_data,batch_loaders,self.args)
                result = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}

                if self.args.eval_verbose:
                    print(f'------------------------------[Seed {seed}]-----------------------------------')
                    edge_evaluation_printer(train_metrics, val_metrics, test_metrics)
                    print(f'------------------------------------------------------------------------------')

                metrics_dict = aggr_metrics(metrics_dict,result)
                continue
            
            self.args.embedding_mode = True 
            encoder = parse_model(self.args,train_data) 
            
            if self.args.aggr_mode=='maxmin':
                aggregator = MaxminAggregator(self.args) 
            elif self.args.aggr_mode=='mean':
                aggregator = MeanAggregator(self.args) 
            elif self.args.aggr_mode=='max':
                aggregator = MaxAggregator(self.args) 
            
            model = EdgePredictor(encoder,aggregator,self.args)
            if self.args.method == 'TMPHN':
                model.aggregator = model.aggregator.to(self.args.device)
            else:
                model = model.to(self.args.device)
            
            model = self.trainer.training(model,train_data,self.args,seed_split=batch_loaders,task_type='edge_pred')

            if self.args.eval_verbose:
                print(f'------------------------------[Seed {seed}]-----------------------------------')
                result=self.evaluator.evaluate(model,train_data,seed_split=batch_loaders,task_type='edge_pred',verbose=True)
                metrics_dict = aggr_metrics(metrics_dict,result) 
                print(f'------------------------------------------------------------------------------')
            else:
                result=self.evaluator.evaluate(model,train_data,seed_split=batch_loaders,task_type='edge_pred',verbose=True)
                metrics_dict = aggr_metrics(metrics_dict,result) 

        print(f'---------------------------------[Final]--------------------------------------')
        avg_result_printer_edge(metrics_dict)
        print(f'------------------------------------------------------------------------------')

    def build_zero_shot_pretrain_model(self, data):
        if not self.args.pretrain_load_path:
            raise ValueError('--downstream_mode zero_shot requires --pretrain_load_path')

        self.args.embedding_mode = True
        encoder = parse_model(self.args,data)
        model = HypergraphPretrainModel(encoder, embedding_dim=self.args.embedding_hidden)

        try:
            checkpoint = torch.load(self.args.pretrain_load_path, map_location=self.args.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.args.pretrain_load_path, map_location=self.args.device)

        if not isinstance(checkpoint, dict) or "pretrain_model" not in checkpoint:
            raise ValueError('zero_shot edge prediction requires a checkpoint saved by pretrain_main.py with pretrain_model')

        pretrain_args = checkpoint.get("args", {})
        pretrain_tasks = str(pretrain_args.get("pretrain_tasks", ""))
        if pretrain_tasks and "fill" not in pretrain_tasks.lower():
            print("Warning: checkpoint pretrain_tasks does not include fill; zero-shot hyperedge scores may be untrained.")

        target_state = model.state_dict()
        source_state = checkpoint["pretrain_model"]
        matched_state = {
            key: value
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        missing_fill_keys = sorted(key for key in target_state if key.startswith("fill_head.") and key not in matched_state)
        if missing_fill_keys:
            raise ValueError(f'checkpoint is missing fill_head parameters required for zero_shot: {missing_fill_keys}')

        incompatible = model.load_state_dict(matched_state, strict=False)
        print(f'Loaded pretrained fill model from {self.args.pretrain_load_path}')
        if incompatible.missing_keys:
            print(f'Missing pretrain-model keys: {incompatible.missing_keys}')
        if incompatible.unexpected_keys:
            print(f'Unexpected pretrain-model keys: {incompatible.unexpected_keys}')

        if self.args.method != 'TMPHN':
            model = model.to(self.args.device)

        return model

    def node_cls_train_eval(self,data):
        
        metrics_dict=defaultdict(list)

        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            masks=data.generate_random_split(train_ratio=self.args.train_prop,val_ratio=self.args.valid_prop,seed=seed)

            if self.args.is_perturbed:
                if self.args.pert_mode in ['spar_label','flip_label']:
                    if self.args.pert_mode == 'spar_label':
                        masks = perturbation(data,mode=self.args.pert_mode,p=self.args.pert_p,masks=masks)
                    elif self.args.pert_mode == 'flip_label':
                        data = perturbation(data,mode=self.args.pert_mode,p=self.args.pert_p,masks=masks)
                    else:
                        raise ValueError('Unimplemented perturbation mode for label robustness')

            if self.args.downstream_mode in ['encoder_finetune','linear_probe'] or self.args.pretrain_load_path:
                if not self.args.pretrain_load_path:
                    if self.args.downstream_mode == 'linear_probe':
                        raise ValueError('--downstream_mode linear_probe requires --pretrain_load_path')
                self.args.embedding_mode = True
                encoder = parse_model(self.args,data)
                model = NodePredictor(encoder,data.num_classes,self.args)
            else:
                self.args.embedding_mode = False
                model = parse_model(self.args,data)

            if self.args.method == 'TMPHN':
                pass
            else:
                model = model.to(self.args.device)

            self.trainer.training(model,data,self.args,seed_split=masks,task_type='node_cls')
            
            self.train_times.append(self.trainer.train_time)

            # Evasion Attack
            if self.args.is_perturbed and not self.args.is_poison:
                test_data = data.evasion_data
            else:
                test_data = data

            if self.args.eval_verbose:
                print(f'------------------------------[Seed {seed}]-----------------------------------')
                result=self.evaluator.evaluate(model,test_data,seed_split=masks,task_type='node_cls',verbose=True)
                print(f'------------------------------------------------------------------------------')
            else:
                result=self.evaluator.evaluate(model,test_data,seed_split=masks,task_type='node_cls',verbose=False)
            
            for m in result:
                metrics_dict[m].append(result[m])
            
        print(f'---------------------------------[Final]--------------------------------------')
        self.test_dict = defaultdict(list) 
        for m in metrics_dict:
            result_printer(metrics_dict[m],m)
            metrics_mean, metrics_std = mean_std_metrics(metrics_dict[m])
            self.test_dict[m].extend([metrics_mean[-1],metrics_std[-1]])
        print(f'Avg Training Time: {np.mean(self.train_times):2f}')
        print(f'------------------------------------------------------------------------------')

    def hg_cls_train_eval(self,data):
        
        metrics_dict=defaultdict(list)
        
        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            train_set,val_set,test_set = generate_split_hypergraphs(data,self.args.train_prop,self.args.valid_prop,seed)
            batch_loaders = generate_hg_loaders(train_set,val_set,test_set,self.args)
            
            self.args.embedding_mode = True 
            encoder = parse_model(self.args,data)

            model = HyperGPredictor(encoder,data.num_classes,self.args)
            if self.args.method == 'TMPHN':
                model.classifer = model.aggregator.to(self.args.device)
            else:
                model = model.to(self.args.device)
                     
            model = self.trainer.training(model,data,self.args,seed_split=batch_loaders,task_type='hg_cls')
            
            if self.args.eval_verbose:
                print(f'------------------------------[Seed {seed}]-----------------------------------')
                result=self.evaluator.evaluate(model,data,seed_split=batch_loaders,task_type='hg_cls',verbose=True)
                print(f'------------------------------------------------------------------------------')
            else:
                result=self.evaluator.evaluate(model,data,seed_split=batch_loaders,task_type='hg_cls',verbose=False)

            for m in result:
                metrics_dict[m].append(result[m])

        print(f'---------------------------------[Final]--------------------------------------')
        for m in metrics_dict:
            result_printer(metrics_dict[m],m)
        print(f'------------------------------------------------------------------------------')
        
    def running(self,task_type,data):
        if self.args.downstream_mode == 'zero_shot' and task_type != 'edge_pred':
            raise ValueError('--downstream_mode zero_shot is currently implemented only for edge_pred. Use finetune or add a linear-probe head for classification tasks.')
        
        if task_type == 'node_cls':
            self.node_cls_train_eval(data)
        elif task_type == 'edge_pred':
            self.edge_pred_train_eval(data)
        elif task_type == 'hg_cls':
            self.hg_cls_train_eval(data)
        else:
            raise NotImplementedError

def build_edge_prediction_graph(data, data_dict, args):
    edge_data = copy.deepcopy(data)
    train_hyperedges = split_positive_hyperedges(data_dict, "train")
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
    
    if args.embedding_mode:
        num_targets=args.embedding_hidden
    else:
        num_targets=data.num_classes
    
    # --------- Hypergraph Semi-supervised Models --------------------
    
    if args.method == 'AllSetformer':
        if args.LearnMask:
            model = SetGNN(data.num_features, num_targets, args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, args)
    elif args.method == 'AllDeepSets':
        args.PMA = False 
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(data.num_features, num_targets, args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, args)
    elif args.method in ['HGNN','HCHA']:
        model = HCHA(data.num_features, num_targets, args)
    elif args.method == 'HNHN':
        model = HNHN(data.num_features, num_targets, args)
    elif args.method in ['UniGIN']:
        model = UniGNN(data.num_features, num_targets, args)
    elif args.method == 'UniGCNII':
        model = UniGCNII(data.num_features, num_targets, args)
    elif args.method == 'HyperGCN':
        model = HyperGCN(data.num_features, num_targets, args)
    elif args.method == 'LEGCN':
        model = LEGCN(data.num_features, num_targets, args)
    elif args.method == 'HJRL':
        model = HJRL(data.num_features, num_targets, args)
    elif args.method == 'HyperND':
        model = HyperND(data.num_features, num_targets, args)
    elif args.method == 'EDHNN':
        model = EquivSetGNN(data.num_features, num_targets, args)
    elif args.method == 'SheafHyperGNN':
        model = SheafHyperGNN(data.num_features,num_targets,args)
    elif args.method == 'EHNN':
        model = EHNN(data.num_features,num_targets,args,data.ehnn_cache)
    elif args.method == 'TMPHN':
        model = TMPHN(data.num_features,num_targets,data.x,data.neig_dict,args)
    elif args.method == 'PhenomNNS':
        model = PhenomNNS(data.num_features,num_targets,args)
    elif args.method == 'PhenomNN':
        model = PhenomNN(data.num_features,num_targets,args)
    elif args.method == 'DPHGNN':
        model = DPHGNN(data.num_features,num_targets,args)
    elif args.method == 'PlainUnigencoder':
        model = PlainUnigencoder(data.num_features, num_targets, args)
    elif args.method == 'TFHNN':
        model = TFHNN(data.num_features,num_targets,args)
    elif args.method == 'MLP':
        model = PlainMLP(data.num_features,num_targets,args)
    elif args.method == 'HyperGT':
        model = HyperGT(data.num_features,num_targets,args)
    elif args.method == 'CEGCN':
        model = CEGCN(data.num_features,num_targets,args)
    elif args.method == 'CEGAT':
        model = CEGAT(data.num_features,num_targets,args)
    else:
        raise ValueError('Unimplemented model')

    return model
