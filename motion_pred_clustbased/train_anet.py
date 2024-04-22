from .trainers.vae_clust_based import VAEClusTrainer
from .trainers.gan_clust_based import GANClusTrainer
from .trainers.anet import AneTrainer

from .utils.common import *

from argparse import ArgumentParser


parser = ArgumentParser(description="Train DL-based forecasters on 2D motion data")

parser.add_argument(
    "--cfg",
    type=str,
    default="motion_pred_clustbased/cfg/training/thor/ours_anet_based.yaml",
    required=False,
    help="configuration file comprising: networks design choices, hyperparameters, etc.",
)

args = parser.parse_args()
cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/training/default.yaml")
model_name = cfg["model"]

# Get clustering and cvae in mode recognizer, recognizer
if model_name == "vae-clust-based":
    t = VAEClusTrainer(cfg)
    metrics, clusterer, gen_model = t.train()
elif model_name == "gan-clust-based":
    t = GANClusTrainer(cfg)
    metrics, clusterer, gen_model = t.train()
else:
    raise NotImplementedError(model_name)

# Train Anet
t_anet = AneTrainer(cfg)
results, anet = t_anet.train(clusterer, gen_model, cfg["anet"])
