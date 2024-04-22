import os
import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import yaml

class Plotter:
    def __init__(self,
                 cfg,
                 subplots = (3, 3)) -> None:
        self.subplots = subplots
        self.n_samples = cfg["n_samples"]
        self.type = cfg["data"]
        str_fn = f"/pp_skip{cfg['skip']}_minped{cfg['min_ped']}"
        target_dataset = str(cfg["dataset"]) + str_fn
        root_dir = cfg["root_dir"]

        self.dir = os.path.join(root_dir, target_dataset)
        self.loading_files()
        self.assign_vars()

    def loading_files(self):
        list_files = os.listdir(self.dir)
        self.data = {}
        for file in list_files:
            fp = os.path.join(self.dir, file)
            with open(fp, "rb") as f:
                self.data[file] = cPickle.load(f)


    def assign_vars(self):
        self.raw_traj = np.stack(self.data["train_raw.pkl"]) # raw trajectories
        self.vect_dir_traj = np.stack(self.data["train_vect_dir.pkl"]) # vector director
        self.dirpolar_traj = np.stack(self.data["train_dirpolar.pkl"]) # vector director in polar coordinates

        # normalized versions
        self.norm_traj = np.stack(self.data["train_norm.pkl"] if self.type == "benchmark" else [self.data["train_norm.pkl"][i][0] for i in range(len(self.data["train_norm.pkl"]))])
        self.vectdir_norm_traj = np.stack(self.data["train_vect_dirnorm.pkl"] if self.type == "benchmark" else [self.data["train_vect_dirnorm.pkl"][i][0] for i in range(len(self.data["train_vect_dirnorm.pkl"]))])
        self.dirpolar_norm_traj = np.stack(self.data["train_dirpolar_norm.pkl"] if self.type == "benchmark" else [self.data["train_dirpolar_norm.pkl"][i][0] for i in range(len(self.data["train_dirpolar_norm.pkl"]))])

        # standard scaler
        self.xmean, self.xscale, self.ymean, self.yscale = self.data["info_norm.pkl"]
        self.xmean_rel, self.xscale_rel, self.ymean_rel, self.yscale_rel = self.data["info_vect_dir.pkl"]
        self.drmean, self.drscale, self.dtmean, self.dtscale = self.data["info_polar_dir.pkl"]


    def plot(self,):
        idxs = np.random.randint(0, len(self.raw_traj), self.n_samples)
        _, ax = plt.subplots(*self.subplots)

        ax[0, 0].set_title(r"$(x, y)$")
        ax[0, 1].set_title(r"$(dx, dy)$")
        ax[0, 2].set_title(r"$(dR, d\theta)$")
        ax[1, 0].set_title(r"n$(x, y)$")
        ax[1, 1].set_title(r"n$(dx, dy)$")
        ax[1, 2].set_title(r"n$(dR, d\theta)$")
        ax[2, 0].set_title(r"$(x, y)'$")
        ax[2, 1].set_title(r"$(dx, dy)'$")
        ax[2, 2].set_title(r"$(dR, d\theta)'$")
        raw_trajs, vect_dir_dens, dirpolar_dens = [], [], []
        for t in idxs:
            raw_trajs.append(np.stack([self.raw_traj[t, :, 0], self.raw_traj[t, :, 1]], axis = -1))
            ax[0, 0].plot(self.raw_traj[t, :, 0], self.raw_traj[t, :, 1], linewidth = 1)
            ax[0, 1].scatter(self.vect_dir_traj[t, :, 0], self.vect_dir_traj[t, :, 1], s = 1)
            ax[0, 2].scatter(self.dirpolar_traj[t, :, 0], self.dirpolar_traj[t, :, 1], s = 1)
            ax[1, 0].plot(self.norm_traj[t, :, 0], self.norm_traj[t, :, 1], linewidth = 1)
            ax[1, 1].scatter(self.vectdir_norm_traj[t, :, 0], self.vectdir_norm_traj[t, :, 1], s = 1)
            ax[1, 2].scatter(self.dirpolar_norm_traj[t, :, 0], self.dirpolar_norm_traj[t, :, 1], s = 1)

            x_raw_denorm = (self.xscale * self.norm_traj[t, :, 0]) + self.xmean
            y_raw_denorm = (self.yscale * self.norm_traj[t, :, 1]) + self.ymean
            assert (np.round(x_raw_denorm, decimals = 4) == np.round(self.raw_traj[t, :, 0], decimals = 4)).all() and (np.round(y_raw_denorm, decimals = 4) == np.round(self.raw_traj[t, :, 1], decimals = 4)).all(), "Problem w/ raw normalization"
            ax[2, 0].plot(x_raw_denorm, y_raw_denorm, linewidth = 1)

            x_vect_dir_denorm = (self.xscale_rel * self.vectdir_norm_traj[t, :, 0]) + self.xmean_rel
            y_vect_dir_denorm = (self.yscale_rel * self.vectdir_norm_traj[t, :, 1]) + self.ymean_rel
            assert (np.round(x_vect_dir_denorm, decimals = 4) == np.round(self.vect_dir_traj[t, :, 0], decimals = 4)).all() and (np.round(y_vect_dir_denorm, decimals = 4) == np.round(self.vect_dir_traj[t, :, 1], decimals = 4)).all(), "Problem w/ dir normalization"
            ax[2, 1].scatter(x_vect_dir_denorm, y_vect_dir_denorm, s = 1)
            vect_dir_dens.append(np.stack([x_vect_dir_denorm, y_vect_dir_denorm], axis = -1))

            dr_polar_denorm = (self.drscale * self.dirpolar_norm_traj[t, :, 0]) + self.drmean
            dt_polar_denorm = (self.dtscale * self.dirpolar_norm_traj[t, :, 1]) + self.dtmean
            assert (np.round(dr_polar_denorm, decimals = 4) == np.round(self.dirpolar_traj[t, :, 0], decimals = 4)).all() and (np.round(dt_polar_denorm, decimals = 4) == np.round(self.dirpolar_traj[t, :, 1], decimals = 4)).all(), "Problem w/ dir polar normalization"
            ax[2, 2].scatter(dr_polar_denorm, dt_polar_denorm, s = 1)
            dirpolar_dens.append(np.stack([dr_polar_denorm, dt_polar_denorm], axis = -1))
        plt.show()
        raw_trajs = np.stack(raw_trajs)
        vect_dir_dens = np.stack(vect_dir_dens)
        dirpolar_dens = np.stack(dirpolar_dens)
        vect_dir_dens[:,0,:] += raw_trajs[:, 0, :]
        vect_dir_dens_x = np.cumsum(vect_dir_dens[:,:, 0], axis = -1)
        vect_dir_dens_y = np.cumsum(vect_dir_dens[:,:, 1], axis = -1)
        raw_f_vect_d = np.stack([vect_dir_dens_x, vect_dir_dens_y], axis = -1)
        assert (np.round(raw_f_vect_d, decimals = 4) == np.round(raw_trajs, decimals = 4)).all(), "Something wrong with the inverse transformation of norm increments"


        x_raw_f_polar = dirpolar_dens[:,:,0] * np.cos(dirpolar_dens[:,:,1]) # Rcos\theta
        y_raw_f_polar = dirpolar_dens[:,:,0] * np.sin(dirpolar_dens[:,:,1]) # Rsin\theta
        x_raw_f_polar[:, 0] += raw_trajs[:, 0, 0]
        y_raw_f_polar[:, 0] += raw_trajs[:, 0, 1]
        x_raw_f_polar = np.cumsum(x_raw_f_polar, axis = -1)
        y_raw_f_polar = np.cumsum(y_raw_f_polar, axis = -1)
        raw_f_polar = np.stack([x_raw_f_polar, y_raw_f_polar], axis = -1)
        assert (np.round(raw_f_polar, decimals = 4) == np.round(raw_trajs, decimals = 4)).all(), "Something wrong with the inverse transformation of polar coords"
        print("Passed all assertions!!")

if __name__ == "__main__":
    parser = ArgumentParser(description = "Assert transformations from the input features")
    parser.add_argument(
        "--cfg",
        type = str,
        default = "motion_pred_clustbased/cfg/data_analysis/thor.yaml",
        required = False,
        help = "configuration file comprising visualization features"
    )

    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    p = Plotter(cfg)
    p.plot()
