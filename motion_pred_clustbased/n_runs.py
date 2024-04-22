import ray
from argparse import ArgumentParser
import numpy as np
import datetime

from .trainers.deterministic import Trainer
from .trainers.gan_based import GanTrainer
from .trainers.vae_clust_based import VAEClusTrainer
from .trainers.gan_clust_based import GANClusTrainer
from .trainers.vae_based import VAETrainer
from .utils.common import *

ray.init()


@ray.remote
def train_model(trainer):
    return trainer.train()


parser = ArgumentParser(description="Train n times forecasters on 2D motion data")

parser.add_argument(
    "--cfg",
    type=str,
    default="motion_pred_clustbased/cfg/training/thor/van_det.yaml",
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

model_name = cfg["model"]

if model_name == "lstm":
    t = Trainer(cfg)
elif model_name in ["sup-cgan", "van-gan"]:  # generative models
    t = GanTrainer(cfg)
elif model_name in ["sup-cvae", "van-vae"]:
    t = VAETrainer(cfg)
elif model_name == "vae-clust-based":
    t = VAEClusTrainer(cfg)
elif model_name == "gan-clust-based":
    t = GANClusTrainer(cfg)
else:
    raise NotImplementedError(model_name)

ts = str(datetime.datetime.now().strftime("%y-%m-%d_%a_%H:%M:%S"))

trainers = [t for _ in range(args.n_runs)]
outputs = ray.get([train_model.remote(tr) for tr in trainers])

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
    os.path.join(
        cfg["save"]["path"], "outputs", f"{model_name}_n_runs_avg_results_{ts}.txt"
    ),
    "w",
)
results_line = [f"{k}: {metric}\n" for k, metric in new_res.items()]
file.writelines(results_line)
file.close()
