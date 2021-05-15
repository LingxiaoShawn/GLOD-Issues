import torch, numpy as np
from grakel.kernels import WeisfeilerLehman, VertexHistogram, Propagation, ShortestPath, PropagationAttr
from sklearn.svm import OneClassSVM    
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, average_precision_score

# from similarity_forest import IsolationSimilarityForest # modified from https://github.com/sfczekalski/similarity_forest/blob/master/simforest/outliers/isolation_simforest.py

class KernelBasedGLAD:
    def __init__(self, kernel, detector, labeled=True,
                       WL_iter=5, PK_bin_width=1,
                       LOF_n_neighbors=20, LOF_n_leaf=30, **kwargs):
        kernels = {
            'WL': WeisfeilerLehman(n_iter=WL_iter, normalize=True, base_graph_kernel=VertexHistogram),
            'PK': Propagation(t_max=WL_iter, w=PK_bin_width, normalize=True) if labeled else 
                  PropagationAttr(t_max=WL_iter, w=PK_bin_width, normalize=True),
        }
        detectors = {
            'OCSVM': OneClassSVM(kernel='precomputed', nu=0.1),
            'LOF': LocalOutlierFactor(n_neighbors=LOF_n_neighbors, leaf_size=LOF_n_leaf, 
                                      metric='precomputed', contamination=0.1),
            # 'IF': current similarity forest also has problem
        }

        assert kernel in kernels.keys()
        assert detector in detectors.keys()
        
        self.kernel = kernels[kernel]
        self.detector = detectors[detector]
        self.kernel_name = kernel
        self.detector_name = detector
        self.labeled = labeled

    def __call__(self, dataset):
        # for inference, output anomaly score
        dataset = to_grakel_dataset(dataset, self.labeled)
        kernel_matrix = self.kernel.fit_transform(dataset)
        
        if self.detector_name == 'OCSVM':
            self.detector.fit(kernel_matrix)
            anomaly_scores = -self.detector.decision_function(kernel_matrix)
        else:
            self.detector.fit(np.amax(kernel_matrix) - kernel_matrix)
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
    
    def get_kernel_matrices(self, normalize=True):
        if self.kernel_name == 'PK':
            return _calculate_kernel_matrix_pk(self.kernel, normalize=normalize)
        if self.kernel_name == 'WL':
            return _calculate_kernel_matrix_wl(self.kernel, normalize=normalize)
    
    def fit_kernel_matrix(self, kernel_matrix, ys):
        if self.detector_name == 'OCSVM':
            self.detector.fit(kernel_matrix)
            anomaly_scores = -self.detector.decision_function(kernel_matrix)
        else:
            self.detector.fit(np.amax(kernel_matrix) - kernel_matrix)
            anomaly_scores = -self.detector.negative_outlier_factor_
            
        roc_auc = roc_auc_score(ys, anomaly_scores)
        pr_auc = average_precision_score(ys, anomaly_scores)
        avg_score_normal = anomaly_scores[ys==0].mean()
        avg_score_abnormal = anomaly_scores[ys==1].mean() 
        
        metrics = {'roc_auc': roc_auc, 
                   'pr_auc': pr_auc, 
                   'average_score_normal': avg_score_normal,
                   'average_score_anomaly': avg_score_abnormal}
        return metrics
        
    def fit_kernel_matrices(self, kernel_matrices, ys):
        results = []
        for kernel_matrix in kernel_matrices:
            results.append(self.fit_kernel_matrix(kernel_matrix, ys))
            
        return results
             
def to_grakel_dataset(dataset, labeled=True):
    def to_grakel_graph(data, labeled=True):
        edges = {tuple(edge) for edge in data.edge_index.T.numpy()}
        if labeled:
            labels = {i: fea.argmax().item()+1 for i, fea in enumerate(data.x)}
        else:
            labels = {i: fea.numpy() for i, fea in enumerate(data.x)}
        return [edges, labels] 
    return [to_grakel_graph(data, labeled) for data in dataset]
    
def _calculate_kernel_matrix_pk(kernel, normalize=True):
    def pairwise_operation(x, y, kernel):
        return np.array([kernel.metric(x[t], y[t]) for t in range(kernel.t_max)])
    
    X = kernel.X
    kernel_matrices = np.zeros(shape=(kernel.t_max, len(X), len(X)))
    cache = list()
    for (i, x) in enumerate(X):
        kernel_matrices[:,i,i] = pairwise_operation(x, x, kernel)
        for (j, y) in enumerate(cache):
            kernel_matrices[:,j,i]= pairwise_operation(y, x, kernel)
        cache.append(x) 
    for i in range(kernel.t_max):
        kernel_matrices[i] = np.triu(kernel_matrices[i]) + np.triu(kernel_matrices[i], 1).T 
        
    accumulative_kernel_matrices = np.add.accumulate(kernel_matrices, 0)
      
    if normalize:
        for i in range(kernel.t_max):
            _X_diag = np.diagonal(kernel_matrices[i])
            kernel_matrices[i] = kernel_matrices[i] / np.sqrt(np.outer(_X_diag, _X_diag))
            
            _X_diag = np.diagonal(accumulative_kernel_matrices[i])
            accumulative_kernel_matrices[i] = accumulative_kernel_matrices[i] / np.sqrt(np.outer(_X_diag, _X_diag))
            
    return kernel_matrices, accumulative_kernel_matrices

def _calculate_kernel_matrix_wl(kernel, normalize=True):
    base_kernels = kernel.X # length = wl-iteration
    n_wl_iters = len(base_kernels)
    kernel_matrices = np.stack([base_kernels[i]._calculate_kernel_matrix() for i in range(n_wl_iters)],
                               axis=0).astype(float) # unormalized 
    accumulative_kernel_matrices = np.add.accumulate(kernel_matrices, 0)
    
    if normalize:
        for i in range(n_wl_iters):
            _X_diag = np.diagonal(kernel_matrices[i]) + 1e-6
            kernel_matrices[i] = kernel_matrices[i] / np.sqrt(np.outer(_X_diag, _X_diag))
            
            _X_diag = np.diagonal(accumulative_kernel_matrices[i]) + 1e-6
            accumulative_kernel_matrices[i] = accumulative_kernel_matrices[i] / np.sqrt(np.outer(_X_diag, _X_diag))
    return kernel_matrices, accumulative_kernel_matrices
               