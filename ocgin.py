import torch
import pytorch_lightning as pl # for training wrapper
from sklearn.metrics import roc_auc_score, average_precision_score

class DeepGLAD(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, weight_decay=5e-4, **kwargs):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay 

    def forward(self, data):
        # generate anomaly score for each graph
        raise NotImplementedError 


    def validation_step(self, batch, batch_idx): 
        if self.current_epoch > 0:
            return self(batch), batch.y#.squeeze(-1)

    def validation_epoch_end(self, outputs):
        if self.current_epoch > 0:
            # assume label 1 is anomaly and 0 is normal. (pos=anomaly, neg=normal)
            anomaly_scores = torch.cat([out[0] for out in outputs]).cpu().detach()
            ys = torch.cat([out[1] for out in outputs]).cpu().detach()
            # import pdb; pdb.set_trace()

            roc_auc = roc_auc_score(ys, anomaly_scores)
            pr_auc = average_precision_score(ys, anomaly_scores)
            avg_score_normal = anomaly_scores[ys==0].mean()
            avg_score_abnormal = anomaly_scores[ys==1].mean()  

            metrics = {'val_roc_auc': roc_auc, 
                   'val_pr_auc': pr_auc, 
                   'val_average_score_normal': avg_score_normal,
                   'val_average_score_anomaly': avg_score_abnormal}
            self.log_dict(metrics)
        
    def test_step(self, batch, batch_idx): 
        return self(batch), batch.y#.squeeze(-1)

    def test_epoch_end(self, outputs):
        # assume label 1 is anomaly and 0 is normal. (pos=anomaly, neg=normal)
        anomaly_scores = torch.cat([out[0] for out in outputs]).cpu().detach()
        ys = torch.cat([out[1] for out in outputs]).cpu().detach()
        # import pdb; pdb.set_trace()


        roc_auc = roc_auc_score(ys, anomaly_scores)
        pr_auc = average_precision_score(ys, anomaly_scores)
        avg_score_normal = anomaly_scores[ys==0].mean()
        avg_score_abnormal = anomaly_scores[ys==1].mean()  

        metrics = {'roc_auc': roc_auc, 
               'pr_auc': pr_auc, 
               'average_score_normal': avg_score_normal,
               'average_score_anomaly': avg_score_abnormal}
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), 
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

class GIN(nn.Module):
    """
    Note: batch normalization can prevent divergence maybe, take care of this later. 
    """
    def __init__(self,  nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False, **kwargs):
        super(GIN, self).__init__()
        self.norm = BatchNorm1d
        self.nlayer = nlayer
        self.act = act
        self.transform = Sequential(Linear(nfeat, nhid), self.norm(nhid))
        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.nns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(nlayer):
            self.nns.append(Sequential(Linear(nhid, nhid, bias=bias), 
                                       act, Linear(nhid, nhid, bias=bias)))
            self.convs.append(GINConv(self.nns[-1]))
            self.bns.append(self.norm(nhid))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.transform(x) # weird as normalization is applying to all ndoes in database
        # maybe a better way is to normalize the mean of each graph, and then apply tranformation
        # to each groups * 
        embed = self.pooling(x, batch)
        std = torch.sqrt(self.pooling((x - embed[batch])**2, batch))
        graph_embeds = [embed]
        graph_stds = [std]
        # can I also record the distance to center, which is the variance?
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            x = self.bns[i](x)
            embed = self.pooling(x, batch) # embed is the center of nodes
            std = torch.sqrt(self.pooling((x - embed[batch])**2, batch))
            graph_embeds.append(embed)
            graph_stds.append(std)

        graph_embeds = torch.stack(graph_embeds)
        graph_stds = torch.stack(graph_stds)

        return graph_embeds, graph_stds


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from test_tube import HyperOptArgumentParser

class OCGIN(DeepGLAD):
    def __init__(self, nfeat,
                 nhid=128, 
                 nlayer=3,
                 dropout=0, 
                 learning_rate=0.001,
                 weight_decay=0,
                 **kwargs):
        model = GIN(nfeat, nhid, nlayer=nlayer, dropout=dropout)
        super().__init__(model, learning_rate, weight_decay)
        self.save_hyperparameters() # self.hparams
        self.radius = 0
        self.nu = 1
        self.eps = 0.01
        self.mode = 'sum' 
        assert self.mode in ['concat', 'sum']
        self.register_buffer('center', torch.zeros(nhid if self.mode=='sum' else (nlayer+1)*nhid ))
        self.register_buffer('all_layer_centers', torch.zeros(nlayer+1, nhid))
        
    def get_hiddens(self, data):
        embs, stds = self.model(data)
        return embs

    def forward(self, data):
        embs, stds = self.model(data)
        if self.mode == 'concat':
            embs = torch.cat([emb for emb in embs], dim=-1) 
        else:
            # sum is used by original GIN method
            embs = embs.sum(dim=0) 

        dist = torch.sum((embs - self.center) ** 2, dim=1)
        scores = dist - self.radius ** 2
        return scores

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            # init the center (and radius) 
            embs, stds = self.model(batch) # for all nodes in the batch
            loss = torch.zeros(1, requires_grad=True, device=self.device) # don't update
            return {'loss':loss, 'emb':embs.detach()}
        else:
            assert self.center != None
            scores = self(batch)
            loss = self.radius ** 2 + (1 / self.nu) * torch.mean(F.relu(scores))
            self.log('training_loss', loss)
            return loss

    def training_epoch_end(self, outputs):
        if self.current_epoch == 0:
            # init center 
            embs = torch.cat([d['emb'] for d in outputs], dim=1)
            self.all_layer_centers = embs.mean(dim=1)
            if self.mode == 'concat':
                self.center = torch.cat([x for x in self.all_layer_centers], dim=-1)
            else:
                # sum is used by original GIN method
                self.center = torch.sum(self.all_layer_centers, 0)
            #self.register_buffer('center', center)
            # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
            # self.center[(abs(self.center) < self.eps) & (self.center < 0)] = -self.eps
            # self.center[(abs(self.center) < self.eps) & (self.center > 0)] = self.eps

    @staticmethod
    def add_model_specific_args(parent_parser):
        # nfeat should be inferred from data later
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)
        # model 
        parser.add_argument('--nhid', type=int, default=32)
        parser.add_argument('--nlayer', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0)
        # optimizer
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        return parser