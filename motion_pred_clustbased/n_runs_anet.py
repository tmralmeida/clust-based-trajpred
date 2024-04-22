import ray
from .trainers.gan_clust_based import GANClusTrainer
from .trainers.vae_clust_based import VAEClusTrainer
from .trainers.anet import AneTrainer

from .utils.common import *

from argparse import ArgumentParser
import numpy as np
import datetime


ray.init()


@ray.remote
def train_anet(cfg):
    if cfg["model"] == "vae-clust-based":
        t = VAEClusTrainer(cfg)
    elif cfg["model"] == "gan-clust-based":
        t = GANClusTrainer(cfg)
    _, clusterer, model = t.train()
    t_anet = AneTrainer(cfg)
    outputs = t_anet.train(clusterer, model, cfg["anet"])
    return outputs


parser = ArgumentParser(description="Train DL-based forecasters on 2D motion data")

parser.add_argument(
    "--cfg",
    type=str,
    default="motion_pred_clustbased/cfg/training/thor/ours_anet_based.yaml",
    required=False,
    help="configuration file comprising: networks design choices, hyperparameters, etc.",
)

parser.add_argument(
    "--n_runs",
    type=int,
    default=5,
    required=False,
    help="Number of runs",
)

args = parser.parse_args()
cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/training/default.yaml")


ts = str(datetime.datetime.now().strftime("%y-%m-%d_%a_%H:%M:%S"))

cfgs = [cfg for _ in range(args.n_runs)]
outputs = ray.get([train_anet.remote(cf) for cf in cfgs])

results = [r[0] for r in outputs]
new_res = results[0]
for dict2 in results[1:]:
    new_res = merge_dicts(new_res, dict2)

print("==================Overall Results================")
print(new_res)
for k, v in new_res.items():
    avg_metric, std_metric = np.mean(v), np.std(v)
    print(f"{k} {avg_metric} +- {std_metric}")


# save txt results file
file = open(
    os.path.join(cfg["save"]["path"], "outputs", f"anet_n_runs_avg_results_{ts}.txt"),
    "w",
)
results_line = [f"{k}: {metric}\n" for k, metric in new_res.items()]
file.writelines(results_line)
file.close()
