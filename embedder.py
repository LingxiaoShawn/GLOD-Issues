import torch
import networkx as nx
from torch_geometric.utils import to_networkx 

from karateclub import Graph2Vec, FGSD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM 
from sklearn.metrics import roc_auc_score, average_precision_score

from test_tube import HyperOptArgumentParser
import logging

class EmbeddingBasedGLAD:
    def __init__(self, embedder, detector, 
                       G2V_nhid=128, G2V_wl_iter=2, 
                       FGSD_hist_bins=200, 
                       IF_n_trees=200, IF_sample_ratio=0.5,
                       LOF_n_neighbors=20, LOF_n_leaf=30, normalize_embedding=False, **kwargs):
        embedders = {
            'Graph2Vec': Graph2Vec(wl_iterations=G2V_wl_iter, dimensions=G2V_nhid, 
                                    attributed=True, epochs=50),
            'FGSD': FGSD(hist_bins=FGSD_hist_bins, hist_range=20)
        }
        detectors = {
            'IF': IsolationForest(n_estimators=IF_n_trees, max_samples=IF_sample_ratio, contamination=0.1),
            'LOF': LocalOutlierFactor(n_neighbors=LOF_n_neighbors, leaf_size=LOF_n_leaf, contamination=0.1),
            'OCSVM': OneClassSVM(gamma='scale', nu=0.1)

        }

        assert embedder in embedders.keys()
        assert detector in detectors.keys()

        self.embedder = embedders[embedder]
        self.detector = detectors[detector]
        self.embedder_name = embedder
        self.detector_name = detector
        self.normalize_embedding = normalize_embedding

    def __call__(self, dataset):
        # for inference, output anomaly score
        dataset = to_networkx_dataset(dataset)
        self.embedder.fit(dataset)
        graph_embeddings = self.embedder.get_embedding()

        if self.normalize_embedding:
            graph_embeddings = graph_embeddings / np.linalg.norm(graph_embeddings, axis=1, keepdims=True)

        self.detector.fit(graph_embeddings)

        if self.detector_name in ['IF', 'OCSVM'] :
            anomaly_scores = -self.detector.decision_function(graph_embeddings)
        else:
            anomaly_scores = -self.detector.negative_outlier_factor_

        return anomaly_scores

    def fit(self, dataset):
        ys = torch.cat([data.y for data in dataset])
        anomaly_scores = self(dataset)

        roc_auc = roc_auc_score(ys, anomaly_scores)
        pr_auc = average_precision_score(ys, anomaly_scores)
        avg_score_normal = anomaly_scores[ys==0].mean()
        avg_score_abnormal = anomaly_scores[ys==1].mean()  

        metrics = {'roc_auc': roc_auc, 
               'pr_auc': pr_auc, 
               'average_score_normal': avg_score_normal,
               'average_score_anomaly': avg_score_abnormal}
        return metrics

    @staticmethod
    def add_model_specific_args(parent_parser):
        # nfeat should be inferred from data later
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)
        # model 
        parser.add_argument('--embedder', type=str, default='Graph2Vec')
        parser.add_argument('--detector', type=str, default='LOF')
        # Graph2Vec params
        parser.add_argument('--G2V_nhid', type=int, default=128)
        parser.add_argument('--G2V_wl_iter', type=int, default=2)
        # FSGD params
        parser.add_argument('--FGSD_hist_bins', type=int, default=200)
        # IF params
        parser.add_argument('--IF_n_trees', type=int, default=100)
        parser.add_argument('--IF_sample_ratio', type=float, default=0.5)
        # LOF params
        parser.add_argument('--LOF_n_neighbors', type=int, default=20)
        parser.add_argument('--LOF_n_leaf', type=int, default=30)
        return parser


def to_networkx_dataset(dataset):
    return [to_networkx_featured(data) for data in dataset]

def to_networkx_featured(data):
    g = to_networkx(data, to_undirected=True, remove_self_loops=True)
    features = {i: fea.argmax()+1 for i, fea in enumerate(data.x)}
    nx.set_node_attributes(g, values=features, name='feature')
    largest_cc = max(nx.connected_components(g), key=len)
    largest_cc = g.subgraph(largest_cc)
    # change node index
    mapping = {old:new for new, old in enumerate(largest_cc.nodes)}
    largest_cc = nx.relabel_nodes(largest_cc, mapping)
    # option 1: remove the checking condition, and remove isolated nodes
    # option 2: for every graph, add a new node, with node feature as 0
    # and connect all nodes to this single virtual node
    
    return largest_cc # get the largest connected component