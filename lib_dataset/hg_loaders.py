import torch
from torch_geometric.data import DataLoader

def generate_split_hypergraphs(dataset,train_ratio,val_ratio,seed):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size 
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], 
                                                                 generator=torch.Generator().manual_seed(42+seed))
    return train_set,val_set,test_set

def generate_hg_loaders(train_set,val_set,test_set,args):
    batch_size = args.hg_batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    batch_loaders = {'train':train_loader,
                     'val':val_loader,
                     'test':test_loader}
    
    return batch_loaders