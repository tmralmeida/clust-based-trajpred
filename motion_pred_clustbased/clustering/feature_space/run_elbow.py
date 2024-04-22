import ray
from ...trainers.gan_based import GanTrainer
from ...utils.common import *

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import datetime

ray.init()

@ray.remote
def train_scgan(trainer, n_clusters):
    return trainer.train(n_clusters)

parser = ArgumentParser(description = "Run Elbow method on SCGAN")

parser.add_argument(
    "--cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/clustering/feature_space/thor.yaml",
    required = False,
    help = "configuration file comprising: networks design choices, hyperparameters, etc."
)

parser.add_argument(
    "--n_runs",
    type = int,
    default = 5,
    required = False,
    help = "Number of runs",
)

parser.add_argument(
    "--n_clusters",
    type = int,
    default = 25,
    required = False,
    help = "Number max of clusters",
)

args = parser.parse_args()
cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/clustering/feature_space/default.yaml")

assert cfg["model"] == "sc-gan", "Elbow method only possible for clustering-related methods"
assert cfg["generator"]["condition"]["train"] == "recognizer" and cfg["generator"]["condition"]["inference"] == "recognizer", "Elbow method recquires one hot labels during training and inference"


ts = str(datetime.datetime.now().strftime('%y-%m-%d_%a_%H:%M:%S'))
metrics = {
    "ade_avg" : [],
    "ade_std" : [],
    "fde_avg" : [],
    "fde_std" : [],
    "sse_avg" : [],
    "sse_std" : [],
    "clust_score_avg" : [],
    "clust_score_std" : []
}
for n_cl in range(2, args.n_clusters + 1):
    t = GanTrainer(cfg)
    inputs = [(t, n_cl) for _ in range(args.n_runs)]
    outputs = ray.get([train_scgan.remote(t, n_cl) for (t, n_cl) in inputs])
    print(f"==================Overall Results for {n_cl} clusters ================")
    results = [r[0] for r in outputs]
    print(results)
    print("========================Avg Results========================")
    new_res = results[0]
    for dict2 in results[1:]:
        new_res = merge_dicts(new_res, dict2)
    for k, v in new_res.items():
        avg_metric, std_metric = np.mean(v), np.std(v)
        print(f"{k} {avg_metric} +- {std_metric}")
        metrics[f"{k}_avg"].append(avg_metric)
        metrics[f"{k}_std"].append(std_metric)




ade_avg = metrics['ade_avg']
ade_std = metrics['ade_std']
fde_avg = metrics['fde_avg']
fde_std = metrics['fde_std']
sse_avg = metrics['sse_avg']
sse_std = metrics['sse_std']
clust_score_avg = metrics['clust_score_avg']
clust_score_std = metrics['clust_score_std']


cl_algo = "kmeans_selfcondgan"

# save ADE, FDE results
plt.figure();
plt.errorbar(np.arange(2, args.n_clusters + 1), metrics['ade_avg'], metrics['ade_std'], ecolor = "r", elinewidth = 0.8)
plt.xlabel("k clusters");
plt.ylabel("ADE");
plt.savefig(os.path.join(cfg["save"]["path"], "outputs", f"{cl_algo}_elbow_ade{ts}.png"))
#plt.close();
# plt.show();


plt.figure();
plt.errorbar(np.arange(2, args.n_clusters + 1), metrics['fde_avg'], metrics['fde_std'], ecolor = "r", elinewidth = 0.8)
plt.xlabel("k clusters");
plt.ylabel("FDE");
plt.savefig(os.path.join(cfg["save"]["path"], "outputs", f"{cl_algo}_elbow_fde{ts}.png"))
#plt.close();
# plt.show();

# save SSE results
plt.figure();
plt.errorbar(np.arange(2, args.n_clusters + 1), metrics['sse_avg'], metrics['sse_std'], ecolor = "r", elinewidth = 0.8)
plt.xlabel("k clusters");
plt.ylabel("SSE");
plt.savefig(os.path.join(cfg["save"]["path"], "outputs", f"{cl_algo}_elbow_sse{ts}.png"))
#plt.close();
# plt.show();

# save clustering score results
plt.figure();
plt.errorbar(np.arange(2, args.n_clusters + 1), metrics['clust_score_avg'], metrics['clust_score_std'], ecolor = "r", elinewidth = 0.8)
plt.xlabel("k clusters");
plt.ylabel("clustering score");
plt.savefig(os.path.join(cfg["save"]["path"], "outputs", f"{cl_algo}_elbow_clust_score{ts}.png"))
#plt.close();
# plt.show();

# save txt results file
file = open(os.path.join(cfg["save"]["path"], "outputs", f"{cl_algo}_elbow_avg_results_{ts}.txt"),"w")
results_line = [f"avgADE: {ade_avg}\n", f"stdADE: {ade_std}\n", f"avgFDE: {fde_avg}\n", f"stdFDE: {fde_std}\n", \
                f"avgSSEs: {sse_avg}\n", f"stdSSEs: {sse_std}\n", f"avgclust_scores: {clust_score_avg}\n", f"stdclust_scores: {clust_score_std}\n"]
file.writelines(results_line)
file.close()