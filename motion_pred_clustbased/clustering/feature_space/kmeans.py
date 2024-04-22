import torch
import numpy as np
from sklearn.cluster import KMeans
from .base_clustering import BaseClusterer


class Clusterer(BaseClusterer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mapping = list(range(self.k))

    def kmeans_fit_predict(self, features, init='k-means++', n_init=10, algorithm = "lloyd"):
        '''fits kmeans, and returns the predictions of the kmeans'''
        # print('Fitting k-means w data shape', features.shape)
        self.method = KMeans(init=init, n_clusters=self.k,
                             n_init=n_init, algorithm = algorithm).fit(features)
        return self.method.predict(features)


    def get_labels(self, x, y):
        d_features = self.get_features(x).detach().cpu().numpy()
        np_prediction = self.method.predict(d_features)
        permuted_prediction = np.array([self.mapping[x] for x in np_prediction])
        return torch.from_numpy(permuted_prediction).long().cpu()