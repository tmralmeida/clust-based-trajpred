import torch
import hdbscan
import numpy as np

class RawClusterer:
    def __init__(self,
                 algorithm_name,
                 method,
                 ts, 
                 n_clusters = -1, 
                 x_cluster = None):
        self.algorithm = algorithm_name
        self.method = method
        self.time_series = ts
        self.k = n_clusters
        self.x = x_cluster
        self.mapping = list(range(self.k))
        
    
    def get_cluster_batch_features(self):
        x = self.x
        return self.get_features(x)
    
    
    def get_features(self, x):
        out = x.reshape(x.shape[0], -1) if not self.time_series else x
        return out
    
        
    def get_labels(self, x, y):
        x = np.float64(x.numpy()) if isinstance(x, torch.Tensor) else x
        x = self.get_features(x)
        
        if self.algorithm in ["k-means", "k-shape"]:
            y = self.method.predict(x)
        elif self.algorithm == "hdbscan": # not valid if inputting distance matrices 
            y, _ = hdbscan.approximate_predict(self.method, x)
                    
        return torch.from_numpy(y).long()
    
    
    def get_label_distribution(self, x = None):
        '''returns the empirical distributon of clustering'''
        y = self.get_labels(x if x else self.x, None)
        counts = [0] * self.k
        for yi in y:
            counts[yi] += 1
        return counts, y
    
    