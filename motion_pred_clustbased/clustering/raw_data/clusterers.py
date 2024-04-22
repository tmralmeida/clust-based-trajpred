import hdbscan
from tslearn.clustering import TimeSeriesKMeans, KShape
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score, silhouette_score
from tslearn.clustering import silhouette_score as ts_silhouette_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import os 
from collections import Counter

from ..utils import FastDiscreteFrechetMatrix, euclidean
from .base_clustering import RawClusterer

class CustomKClust:
    def __init__(self, 
                 algorithm : str,
                 cfg : dict,
                 writer = None):
        self.algorithm = algorithm
        self.cfg = cfg
        self.max_iter = cfg["max_iter"]
        self.n_init  = cfg["n_init"]
        self.metric = cfg["metric"] if "metric" in cfg.keys() else None
        self.n_jobs =  cfg["n_jobs"] if "n_jobs"  in cfg.keys() else None # for time series kmeans and flat kmeans
        self.init = cfg["init"] if "init"  in cfg.keys() else "random" # for time series kmeans and flat kmeans
        self.elbow = cfg["elbow"]["run"] if "elbow"  in cfg.keys() else None
        self.type = cfg["type"] if algorithm == "k-means" else "time_series" # options time_series, normal, k-shape
        self.writer = writer
        
        
    def ts_kmeans(self, n_clusters : int):
        """ Function to run time series kmeans based on tslearn lib """
        return TimeSeriesKMeans(n_clusters = n_clusters, 
                                max_iter = self.max_iter,
                                n_init = self.n_init,
                                metric = self.metric,
                                n_jobs = self.n_jobs, 
                                random_state = 42,
                                init = self.init)
    
    
    def flat_kmeans(self, n_clusters : int):
        """ Function to run flattened time series kmeans based on sklearn lib """
        return KMeans(n_clusters = n_clusters, 
                      init = self.init,
                      n_init = self.n_init,
                      max_iter = self.max_iter,
                      random_state = 42)
        
        
    def kshape(self, n_clusters : int) :
        """ Function to run k-shape clustering algorithm based on tslearn lib """ 
        return KShape(n_clusters = n_clusters, 
                      max_iter = self.max_iter,
                      random_state = 42,
                      n_init = self.n_init)
        
        
    def run(self, dataset : np.array):
        self.min_cl, self.max_cl = (self.cfg["n_clusters"],self.cfg["n_clusters"]) if not self.elbow \
                                    else (self.cfg["elbow"]["min_clusters"],self.cfg["elbow"]["max_clusters"])
        metrics, clustererers = {}, []
        metrics["sse"], metrics["clust_score"], metrics["N Iterations"] = [], [], []
        for n_cl in range(self.min_cl, self.max_cl + 1):
            print(f"\nRunning {self.type} {self.algorithm} with {n_cl} clusters...")
            if self.type == "time_series":
                ss = ts_silhouette_score
                clusterer = self.ts_kmeans(n_cl) if self.algorithm == "k-means" else self.kshape(n_cl)
            elif self.type == "normal":
                dataset = dataset.reshape(dataset.shape[0], -1)
                clusterer = self.flat_kmeans(n_cl) 
                ss = davies_bouldin_score
            clusterer = clusterer.fit(dataset)
            clusterer = RawClusterer(self.algorithm, clusterer, self.type == "time_series", n_cl, dataset)
            data_labels = clusterer.get_labels(dataset, None)
            clustering_score_avg = ss(dataset, data_labels.numpy())
            sse = clusterer.method.inertia_
            niters = clusterer.method.n_iter_
            print(f"SSE={sse}, Clustering score={clustering_score_avg} in {niters} iterations.")
            if self.elbow:
                metrics["clust_score"].append(clustering_score_avg)
                metrics["sse"].append(sse)
                metrics["N Iterations"].append(niters)
                clustererers.append(clusterer)
            else:
                metrics["clust_score"] = clustering_score_avg
                metrics["sse"] = sse
                metrics["N Iterations"] = niters
                clustererers = clusterer
            if self.writer:
                inp_emb = torch.from_numpy(dataset)
                self.writer.add_embedding(inp_emb.view(inp_emb.size(0), -1), metadata=data_labels, global_step = n_cl, tag = "feature space")
        return metrics, clustererers
        

class CustomDBSCAN:
    def __init__(self, 
                algorithm : str, 
                cfg : dict,
                out_path : str, 
                type : str,
                writer = None):
        self.algorithm = algorithm
        self.cfg = cfg 
        self.n_jobs =  cfg["n_jobs"] if "n_jobs"  in cfg.keys() else None
        self.elbow = cfg["elbow"]["run"] if "elbow"  in cfg.keys() else None
        self.type = type
        self.writer = writer
        self.out_path = out_path
        
        
    def cmpt_eucmat(self, x):
        """ Compute euclidean distance matrix """
        fast_frechet = FastDiscreteFrechetMatrix(euclidean)
        n_traj = len(x)
        dist_mat = np.zeros((n_traj, n_traj), dtype=np.float64)
        for i in range(n_traj - 1):
            p = x[i]
            for j in range(i + 1, n_traj):
                q = x[j]
                dist_mat[i, j] = fast_frechet.distance(p, q)
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat
    
    
    def cmpt_polarmat(self, x):
        """ Compute polar distance matrix """
        fast_frechet = FastDiscreteFrechetMatrix(euclidean)
        n_traj = len(x)
        dist_mat = np.zeros((n_traj, n_traj), dtype=np.float64)
        for i in range(n_traj - 1):
            p_x = x[i][:, 0] * np.cos(x[i][:, 1])
            p_y = x[i][:, 0] * np.sin(x[i][:, 1])
            p = np.stack([p_x, p_y], axis = 1)
            for j in range(i + 1, n_traj):
                q_x = x[j][:, 0] * np.cos(x[j][:, 1])
                q_y = x[j][:, 0] * np.sin(x[j][:, 1])
                q = np.stack([q_x, q_y], axis = 1)
                dist_mat[i, j] = fast_frechet.distance(p, q)
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat
        

        
    def run(self, dataset : np.array):
        # compute distances
        if self.type == "vect_dir":
            dist_mat = self.cmpt_eucmat(dataset)
        elif self.type == "polar":
            dist_mat = self.cmpt_polarmat(dataset)
        else:
            raise NotImplementedError(f"{self.type} distance type")
        n_dims = dataset.shape[-1] * dataset.shape[-2]
        print("Distance matrix computed!")
        if self.elbow:
            neighbors = NearestNeighbors(n_neighbors=n_dims * 2)
            neighbors_fit = neighbors.fit(dist_mat)
            distances, _ = neighbors_fit.kneighbors(dist_mat)
            distances = np.sort(distances, axis=0)
            distances = distances[:,1]
            plt.title(f"{self.algorithm} - Distances from knn")
            plt.plot(distances);
            plt.xlabel("number of points");
            plt.ylabel(f"distances");
            plt.savefig(os.path.join(self.out_path, f"knn_distances_k{n_dims*2}.png"))
            plt.show();
             # input from the user
            eps_min = float(input("Enter eps_min: "))
            eps_max = float(input("Enter max: "))
            step = float(input("Enter step: "))
            epss = np.arange(eps_min,eps_max, step) 
            min_samples = range(self.cfg["elbow"]["min_samples_ratio"]*n_dims, self.cfg["elbow"]["max_samples_ratio"]*n_dims)
        else:
            epss = np.array(self.cfg["eps"])[None,]
            min_samples = range(2*n_dims, 2*n_dims+1) #2* dim https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd
        metrics, output = {}, []
        metrics["min_samples"], metrics["eps"], metrics["Silhouette score"] = [], [], []
        for ms in min_samples:
            for eps in epss:
                print(f"\nRunning dbscan with {ms} min samples and {eps} distance...")
                dbscan = DBSCAN(eps=eps, 
                                min_samples=ms,
                                metric = "precomputed",
                                n_jobs = -1)
                data_labels = dbscan.fit_predict(dist_mat)
                data_labels_tsr = torch.from_numpy(data_labels)
                silhouette_avg = silhouette_score(dist_mat, data_labels)
                print("Silhouette score:", silhouette_avg)
                metrics["min_samples"].append(ms)
                metrics["eps"].append(eps)
                metrics["Silhouette score"].append(silhouette_avg)
                output.append((ms, eps, silhouette_avg))
        min_samples, eps, score = sorted(output, key=lambda x:x[-1])[-1]
        print(f"\n\nBest silhouette_score: {score}")
        print(f"min_samples: {min_samples}")
        print(f"eps: {eps}")
        labels = DBSCAN(min_samples=min_samples, metric = "precomputed", eps = eps, n_jobs = self.n_jobs).fit_predict(dist_mat)
        clusters = len(Counter(labels))
        print(f"Number of clusters: {clusters}")
        print(f"Number of outliers: {Counter(labels)[-1]}")
        print(f"Silhouette_score: {silhouette_score(dist_mat, labels)}")
        if self.writer:
                inp_emb = torch.from_numpy(dataset)
                self.writer.add_embedding(inp_emb.view(inp_emb.size(0), -1), metadata=data_labels_tsr, global_step = clusters, tag = "feature space")
        return metrics, clusters, dbscan
    
    
class CustomHDBSCAN(CustomDBSCAN):
    def __init__(self, algorithm: str, cfg: dict, out_path: str, type: str, writer=None):
        super().__init__(algorithm, cfg, out_path, type, writer)
        self.min_cluster_size = cfg["min_cluster_size"]
        self.min_samples = cfg["min_samples"]
        
    def run(self, dataset : np.array):
        print(f"\nRunning HDBSCAN with {self.min_cluster_size} min cluster size and {self.min_samples} min samples...")
        if self.type == "vect_dir":
            dist_mat = self.cmpt_eucmat(dataset)
        elif self.type == "polar":
            dist_mat = self.cmpt_polarmat(dataset)
        else:
            raise NotImplementedError(f"{self.type} distance type")
        print("Distance matrix computed!")
        metrics = {}
        metrics["min_samples"], metrics["min_cluster_size"], metrics["Silhouette score"] = [], [], []
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, cluster_selection_method='leaf', allow_single_cluster=True)
        cluster_labels = clusterer.fit_predict(dist_mat)
        data_labels_tsr = torch.from_numpy(cluster_labels)
        silhouette_avg = silhouette_score(dist_mat, cluster_labels)
        metrics["min_samples"].append(self.min_samples)
        metrics["min_cluster_size"].append(self.min_cluster_size)
        metrics["Silhouette score"].append(silhouette_avg)
        clusters = len(Counter(cluster_labels))
        print(f"Number of clusters: {clusters}")
        print(f"Number of outliers: {Counter(cluster_labels)[-1]}")
        print(f"Silhouette_score: {silhouette_avg}")
        if self.writer:
                inp_emb = torch.from_numpy(dataset)
                self.writer.add_embedding(inp_emb.view(inp_emb.size(0), -1), metadata=data_labels_tsr, global_step = clusters, tag = "feature space")
        return metrics, clusters, clusterer
        