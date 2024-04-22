from ...utils.common import load_config
from ...trainers.gan_based import GanTrainer

from argparse import ArgumentParser


parser = ArgumentParser(description = "Running sc-gan absed clustering method")

parser.add_argument(
    "--cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/clustering/feature_space/thor.yaml",
    required = False,
    help = "configuration file comprising: networks design choices, hyperparameters, etc."
)

args = parser.parse_args()
cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/clustering/feature_space/default.yaml")
gt = GanTrainer(cfg)
results, best_clusterer, best_generator = gt.train()
print("\n\nResults:", results)