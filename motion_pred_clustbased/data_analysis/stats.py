from argparse import ArgumentParser
import matplotlib
import yaml
import numpy as np
from .assert_transforms import Plotter
import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser(description = "Compute statistics based on the input features")
parser.add_argument(
    "--cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/data_analysis/thor.yaml",
    required = False,
    help = "configuration file comprising dataset settings"
)

args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

class Stats(Plotter):
    def __init__(self, cfg, subplots=(3, 3)) -> None:
        super().__init__(cfg, subplots)

    def compute_avgs(self):
        dx = self.vectdir_norm_traj[:,1:,0]
        dy = self.vectdir_norm_traj[:,1:,1]
        dr = self.dirpolar_norm_traj[:,1:,0]
        dt = self.dirpolar_norm_traj[:,1:,1]
        print("\n Normalized Stats:")
        print(f"(dx, dy): {dx.mean(), dy.mean()}+-{dx.std(), dy.std()}")
        print(f"(dr, dt): {dr.mean(), dt.mean()}+-{dr.std(), dt.std()}\n")


    def compute_si(self):
        last_pt = self.raw_traj[:, -1, :]
        init_pt = self.raw_traj[:, 0, :]
        sd_dist = np.linalg.norm(last_pt - init_pt, axis = -1)
        dists = np.linalg.norm(self.vect_dir_traj, axis = -1).sum(axis = -1) + 1*10**(-6)
        si = sd_dist / dists
        print(f"SI: {si.mean()}+-{si.std()}")

        plt.figure();
        (
            pd.cut(si, bins=10)
                .value_counts()
                .sort_index()
                .plot.bar()
        )
        plt.ylabel("Count");
        plt.xlabel("SI")
        plt.tight_layout();
        plt.show();


if __name__ == "__main__":
    s = Stats(cfg)
    s.compute_si()
    s.compute_avgs()