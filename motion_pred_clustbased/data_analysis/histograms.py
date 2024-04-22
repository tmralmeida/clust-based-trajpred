import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import yaml
import copy
from .assert_transforms import Plotter



parser = ArgumentParser(description = "Visualize histograms comparing the input features")
parser.add_argument(
    "--cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/data_analysis/thor.yaml",
    required = False,
    help = "configuration file comprising histograms features"
)

args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)


class Hists(Plotter):
    def __init__(self, cfg, subplots=(3, 3)) -> None:
        super().__init__(cfg, subplots)
        self.hist_type = cfg["hist_type"]
        self.subplots = (2, 3)


    def __inpfreq_hist(self,):
        cmap = copy.copy(plt.cm.jet)
        cmap.set_bad((0,0,0))  # Fill background with black
        _, ax = plt.subplots(*self.subplots)
        ax[0, 0].set_title(r"$(x, y)$")
        ax[0, 1].set_title(r"$(dx, dy)$")
        ax[0, 2].set_title(r"$(dR, d\theta)$")
        ax[1, 0].set_title(r"n$(x, y)$")
        ax[1, 1].set_title(r"n$(dx, dy)$")
        ax[1, 2].set_title(r"n$(dR, d\theta)$")

        x_raw = self.raw_traj[:,:,0].flatten()
        y_raw = self.raw_traj[:,:,1].flatten()
        x_norm = self.norm_traj[:,:,0].flatten()
        y_norm = self.norm_traj[:,:,1].flatten()
        dx =  self.vect_dir_traj[:,1:, 0].flatten()
        dy = self.vect_dir_traj[:,1:, 1].flatten()
        dx_norm =  self.vectdir_norm_traj[:,1:, 0].flatten()
        dy_norm = self.vectdir_norm_traj[:,1:, 1].flatten()
        r_polar = self.dirpolar_traj[:,1:,0].flatten()
        t_polar = self.dirpolar_traj[:,1:,1].flatten()
        r_polar_norm = self.dirpolar_norm_traj[:,1:,0].flatten()
        t_polar_norm = self.dirpolar_norm_traj[:,1:,1].flatten()


        hist1 = ax[0, 0].hist2d(x_raw, y_raw, bins = 200, norm = LogNorm(), cmap = cmap)
        hist2 = ax[1, 0].hist2d(x_norm, y_norm, bins = 200, norm = LogNorm(), cmap = cmap)
        hist3 = ax[0, 1].hist2d(dx, dy, bins = 200, norm = LogNorm(), cmap = cmap)
        hist4 = ax[1, 1].hist2d(dx_norm, dy_norm, bins = 200, norm = LogNorm(), cmap = cmap)
        hist5 = ax[0, 2].hist2d(r_polar, t_polar, bins = 200, norm = LogNorm(), cmap = cmap)
        hist6 = ax[1, 2].hist2d(r_polar_norm, t_polar_norm, bins = 200, norm = LogNorm(), cmap = cmap)



        plt.colorbar(hist1[-1], shrink = 0.6, ax = ax[0, 0]);
        plt.colorbar(hist2[-1], shrink = 0.6, ax = ax[1, 0]);
        plt.colorbar(hist3[-1], shrink = 0.6, ax = ax[0, 1]);
        plt.colorbar(hist4[-1], shrink = 0.6, ax = ax[1, 1]);
        plt.colorbar(hist5[-1], shrink = 0.6, ax = ax[0, 2]);
        plt.colorbar(hist6[-1], shrink = 0.6, ax = ax[1, 2]);

        plt.show()


    def __cnt_per_interval(self,):
        fig, ax = plt.subplots(1, 3)
        ax[0].set_title(r"$(X, Y) norm$")
        ax[1].set_title(r"$(dX, dY) norm$")
        ax[2].set_title(r"$(dR, d\theta) norm$")


        bins = np.round(np.arange(-1.0, 1.1, 0.1), decimals=2)
        x_norm = self.norm_traj[:,:,0].flatten()
        y_norm = self.norm_traj[:,:,1].flatten()
        dx_norm = self.vectdir_norm_traj[:,1:,0].flatten()
        dy_norm = self.vectdir_norm_traj[:,1:,1].flatten()
        r_norm = self.dirpolar_norm_traj[:,1:,0].flatten()
        t_norm = self.dirpolar_norm_traj[:,1:,1].flatten()
        width = 0.35
        (
            pd.cut(x_norm, bins=bins, include_lowest=True)
                .value_counts()
                .sort_index()
                .plot.bar(ax=ax[0], width = width, alpha=0.5, label = r"$X$")
        )
        (
            pd.cut(y_norm, bins=bins, include_lowest=True)
                .value_counts()
                .sort_index()
                .plot.bar(ax=ax[0], color = "r",  width = width, alpha = 0.5, label = r"$Y$")
        )
        (
            pd.cut(dx_norm, bins=bins, include_lowest=True)
                .value_counts()
                .sort_index()
                .plot.bar(ax=ax[1], width = width, alpha=0.5, label = r"$dX$")
        )
        (
            pd.cut(dy_norm, bins=bins, include_lowest=True)
                .value_counts()
                .sort_index()
                .plot.bar(ax=ax[1], color = "r",  width = width, alpha = 0.5, label = r"$dY$")
        )
        (
            pd.cut(r_norm, bins=bins, include_lowest=True)
                .value_counts()
                .sort_index()
                .plot.bar(ax=ax[2],  width = width, alpha=0.5, label = r"$dR$")
        )
        (
            pd.cut(t_norm, bins=bins, include_lowest=True)
                .value_counts()
                .sort_index()
                .plot.bar(ax=ax[2], color = "r",  width = width, alpha = 0.5, label = r"$d\theta$")
        )

        fig.tight_layout()
        ax[0].legend();
        ax[1].legend();
        ax[2].legend();
        plt.show()


    def __step_wise(self, ):
        coords = [self.norm_traj, self.vectdir_norm_traj[:, 1:, :], self.dirpolar_norm_traj[:, 1:, :]]
        bins = [np.round(np.arange(c.min()-0.1, c.max()+0.1, 0.5), decimals=2) for c in coords]
        lbls = [(r"$X$", r"$Y$"), (r"$dX$", r"$dY$"), (r"$dR$", r"$d\theta$")]
        for k in range(3):
            _, ax = plt.subplots(5, 4)
            ts = 0
            for i in range(5):
                for j in range(3 if i == 4 and k > 0 else 4):
                    ax[i, j].set_title(f"Time Step = {ts}")
                    ts_norm_1 = coords[k][:,ts,0].flatten()
                    ts_norm_2 = coords[k][:,ts,1].flatten()
                    (
                        pd.cut(ts_norm_1, bins=bins[k], include_lowest=True)
                            .value_counts()
                            .sort_index()
                            .plot.bar(ax=ax[i, j], alpha=0.5, label = lbls[k][0])
                    )
                    (
                        pd.cut(ts_norm_2, bins=bins[k], include_lowest=True)
                            .value_counts()
                            .sort_index()
                            .plot.bar(ax=ax[i, j], color = "r", alpha=0.5, label = lbls[k][1])
                    )
                    if i != 4:
                        ax[i, j].get_xaxis().set_visible(False)
                    ax[i, j].legend();

                    ts += 1
        plt.show()

    def __trip_length(self):
        dists = np.linalg.norm(self.vect_dir_traj, axis = -1).sum(axis = -1)
        plt.figure();
        (
            pd.cut(dists, bins=10)
                .value_counts()
                .sort_index()
                .plot.bar()
        )
        plt.ylabel("Count");
        plt.xlabel("Distance (m)")
        plt.tight_layout();
        plt.show();






    def run(self,):
        if self.hist_type == "input_frequency":
            self.__inpfreq_hist()
        elif self.hist_type == "cnt_per_interval":
            self.__cnt_per_interval()
        elif self.hist_type == "step_wise":
            self.__step_wise()
        elif self.hist_type == "trip_length":
            self.__trip_length()
        else:
            raise NotImplementedError(self.hist_type)

if __name__ == "__main__":
    h = Hists(cfg)
    h.run()