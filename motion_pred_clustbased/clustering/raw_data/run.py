from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import _pickle as cPickle
import os
import matplotlib.pyplot as plt

from ...utils.common import *
from ..utils import *
from .clusterers import *

# Check https://neptune.ai/blog/clustering-algorithms out

parser = ArgumentParser(description = "Run clustering methods on raw trajectory data")
parser.add_argument(
    "--cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/clustering/raw/thor.yaml",
    required = False,
    help = "configuration file comprising dataset settings"
)

args = parser.parse_args()
cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/clustering/raw/default.yaml")


# create dirs to save objects
out_path, run_path = create_dirs(cfg)

# create summary writer tb
writer = SummaryWriter(run_path)

clustering_info = cfg["clustering"]
data_info = cfg["data"]

#  loading data
ds = data_info["dataset"]
print(f"\nClustering {ds}...")
skip, min_ped = data_info['skip'], data_info['min_ped']
target_dataset = str(data_info["dataset_target"]) + (f"/pp_skip{skip}_minped{min_ped}" if ds in ["thor", "argoverse"] else f"/pp_skip{skip}_minped{min_ped}")

dir = os.path.join(data_info["data_dir"], target_dataset)
list_files = os.listdir(dir)
data = {}
for file in list_files:
    fp = os.path.join(dir, file)
    with open(fp, "rb") as f:
        data[file] = cPickle.load(f)

norm_traj = np.stack(data["train_norm.pkl"] if ds == "benchmark" else [data["train_norm.pkl"][i][0] for i in range(len(data["train_norm.pkl"]))])
vectdir_norm_traj = np.stack(data["train_vect_dirnorm.pkl"] if ds == "benchmark" else [data["train_vect_dirnorm.pkl"][i][0] for i in range(len(data["train_vect_dirnorm.pkl"]))])
dirpolar_norm_traj = np.stack(data["train_dirpolar_norm.pkl"] if ds == "benchmark" else [data["train_dirpolar_norm.pkl"][i][0] for i in range(len(data["train_dirpolar_norm.pkl"]))])

print("Dataset loaded!")

if data_info["inputs"] == "raw":
    inputs = norm_traj
if data_info["inputs"] == "vect_dir":
    inputs = vectdir_norm_traj[:, 1:, :]
if data_info["inputs"] == "polar":
    inputs = dirpolar_norm_traj[:, 1:, :]

algorithm = clustering_info["algorithm"] # clustering algorithm
cl_hyperparams = clustering_info[algorithm]
if algorithm in ["k-means", "k-shape"]:
    # assert data_info["inputs"] == "raw" or data_info["inputs"] == "vect_dir", f"{algorithm} not prepared for {data_info['inputs']}"
    clust = CustomKClust(algorithm,
                         cl_hyperparams,
                         writer)
    metrics, clustererers = clust.run(inputs)
elif algorithm == "dbscan":
    clust = CustomDBSCAN(algorithm,
                         cl_hyperparams,
                         out_path,
                         data_info["inputs"],
                         writer)
    metrics, clusters, _ = clust.run(inputs)
elif algorithm == "hdbscan":
    clust = CustomHDBSCAN(algorithm,
                          cl_hyperparams,
                          out_path,
                          data_info["inputs"],
                          writer)
    metrics, clusters, _ = clust.run(inputs)
else:
    raise NotImplementedError(algorithm)

save_results(out_path, metrics, clust.max_cl if algorithm in ["k-means", "k-shape"] else clusters)
if cl_hyperparams["elbow"]["run"] and algorithm in ["k-means", "k-shape"]:
    for k, v in metrics.items():
        plt.figure();
        plt.plot(np.arange(clust.min_cl, clust.max_cl+1), v)
        plt.title(f"{clustering_info['algorithm']} - {k}")
        plt.xlabel("k clusters");
        plt.ylabel(f"{k}");
        plt.savefig(os.path.join(out_path, f"elbow_{k}.png"))
        plt.show();


print("Clustering done!")
