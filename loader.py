"""
Data Loaders

Here we create anomaly detection datasets by downsampling from graph classification datasets.  

There are two type of binary graph classification datasets: 
 - Type 1: Class X and Class Not-X
 - Type 2: Class X and Class Y

For binary graph classification dataset, we downsample the down_class as anomaly class.  
It can belong to either type 1 or type 2. 

For multi-class graph classification dataset, we only keep 2 classes to create binary classification dataset,
and downsample the first mentioned class as anomaly class. 

For all graph classification datasets, we assume the dataset is roughly balanced for all classes.
"""
import torch, os
import numpy as np
import random
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, DenseDataLoader
from torch_geometric.utils import degree

# discretrize continous degree
from sklearn.preprocessing import KBinsDiscretizer

DATA_PATH = 'datasets'
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

import torch.nn.functional as F
class OneHotDegree(object):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(self, max_degree, in_degree=False, cat=True):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = truncate_degree(degree(idx, data.num_nodes, dtype=torch.long))
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.max_degree)

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def truncate_degree(degree):
    degree[ (100<=degree) & (degree <200) ] = 101
    degree[ (200<=degree) & (degree <500) ] = 102
    degree[ (500<=degree) & (degree <1000) ] = 103
    degree[ (1000<=degree) & (degree <2000) ] = 104
    degree[ degree >= 2000] = 105
    return degree

class DownsamplingFilter(object):
    def __init__(self, min_nodes, max_nodes, down_class, down_rate, num_classes, second_class=None):
        super(DownsamplingFilter, self).__init__()
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.down_class = down_class
        self.down_rate = down_rate #anomaly/#normal

        self.num_classes = num_classes 
        self.second_class = second_class # for multi-class version 

        if self.num_classes > 2:
            assert second_class is not None 
            assert second_class != down_class
        

    def __call__(self, data):
        # step 1: filter by  graph node size
        keep = (data.num_nodes <= self.max_nodes) and (data.num_nodes >= self.min_nodes)
        # deal with multi-class dataset 
        if self.num_classes > 2:
            keep = keep and (data.y.item() in [self.down_class, self.second_class])
        
        if keep:
            # step 2: downsampling class, treat down_class as anomaly class, and relabel it as 1. 
            anomalous_class = (data.y.item() == self.down_class) 
            data.y.fill_(int(anomalous_class)) # anomalous class as positive
            if anomalous_class:
                # downsampling the anomaly class
                if np.random.rand() > self.down_rate:
                    keep = False
        return keep

def load_data(data_name, down_class=0, down_rate=1, second_class=None, seed=1213, return_raw=False):
    ignore_edge_weight=True
    one_class_train = False
    np.random.seed(seed)
    torch.manual_seed(seed)

    if data_name in ['MNIST', 'CIFAR10']:
        dataset_raw = GNNBenchmarkDataset(root=DATA_PATH, name=data_name)
    else:
        # TUDataset
        use_node_attr = True if data_name == 'FRANKENSTEIN' else False
        dataset_raw = TUDataset(root=DATA_PATH, name=data_name, use_node_attr=use_node_attr)

    if return_raw:
        return dataset_raw

    # downsampling 
    # Get min and max node and filter them, 
    num_nodes_graphs = [data.num_nodes for data in dataset_raw]
    min_nodes, max_nodes = min(num_nodes_graphs), max(num_nodes_graphs)
    if max_nodes >= 10000:
        max_nodes = 10000
    print("min nodes, max nodes:", min_nodes, max_nodes)
    
    # build the filter and transform the dataset
    filter = DownsamplingFilter(min_nodes, max_nodes, down_class, down_rate, dataset_raw.num_classes, second_class)
    indices = [i for i, data in enumerate(dataset_raw) if filter(data)]
    # now down_class is labeled 1, second_class is labeled 0 
    dataset = dataset_raw[torch.tensor(indices)].shuffle() # shuffle the dataset 

    # report the proportion info of the dataset
    labels = np.array([data.y.item() for data in dataset])
    label_dist = ['%d'% (labels==c).sum() for c in range(dataset.num_classes)]
    print("Dataset: %s, Number of graphs: %s [orignal classes: %d, %d], Num of Features %d"%(
            data_name, label_dist, second_class if second_class is not None else 1-down_class, 
            down_class, dataset.num_features))

    # preprocessing: do not use original edge features or weights
    if ignore_edge_weight:
        dataset.data.edge_attr = None

    # deal with no-attribute case
    """
    Another way: discreteize the degree by range. Not always one hot. 
    """
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset_raw: # ATTENTION: use dataset_raw instead of downsampled version!
            degs += [truncate_degree(degree(data.edge_index[0], dtype=torch.long))]
            max_degree = max(max_degree, degs[-1].max().item())
        dataset.transform = OneHotDegree(max_degree)

    # now let's transform in memory before feed into dataloader to save runtime
    dataset_list = [data for data in dataset]

    n = (len(dataset) + 9) // 10
    m = 9 
    train_dataset = dataset_list[:m*n] # 90% train
    val_dataset = dataset_list[m*n:]
    test_dataset = dataset_list

    if one_class_train:
        indices = [i for i, data in enumerate(train_dataset) if data.y.item()==0]
        train_dataset = train_dataset[torch.tensor(indices)] # only keep normal class left

    return train_dataset, val_dataset, test_dataset, dataset, max_nodes


def create_loaders(data_name, batch_size=32, down_class=0, second_class=None, down_rate=1, seed=15213, num_workers=0):

    train_dataset, val_dataset, test_dataset, dataset, max_nodes = load_data(data_name, 
                                                                                down_class=down_class, 
                                                                                second_class=second_class,
                                                                                down_rate=down_rate, 
                                                                                seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset, max_nodes