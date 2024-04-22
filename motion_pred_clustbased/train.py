from .trainers.deterministic import Trainer
from .trainers.gan_based import GanTrainer
from .trainers.vae_based import VAETrainer
from .trainers.vae_clust_based import VAEClusTrainer
from .trainers.gan_clust_based import GANClusTrainer
from .utils.common import *

from argparse import ArgumentParser

parser = ArgumentParser(description="Train DL-based forecasters on 2D motion data")

parser.add_argument(
    "--cfg",
    type=str,
    default="motion_pred_clustbased/cfg/training/thor/van_vae.yaml",
    required=False,
    help="configuration file comprising: networks design choices, hyperparameters, etc.",
)

args = parser.parse_args()
cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/training/default.yaml")
model_name = cfg["model"]

if model_name == "lstm":
    t = Trainer(cfg)
elif model_name in ["sup-cgan", "van-gan"]:  # GAN-based models
    t = GanTrainer(cfg)
elif model_name in ["sup-cvae", "van-vae"]:
    t = VAETrainer(cfg)
elif model_name == "vae-clust-based":
    t = VAEClusTrainer(cfg)
elif model_name == "gan-clust-based":
    t = GANClusTrainer(cfg)
else:
    raise NotImplementedError(model_name)
t.train()
