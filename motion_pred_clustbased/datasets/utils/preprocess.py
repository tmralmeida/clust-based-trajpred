import pandas as pd
import numpy as np
import os
import math
from sklearn.preprocessing import StandardScaler
import _pickle as cPickle
import yaml
from argparse import ArgumentParser


parser = ArgumentParser(description = "Preprocess human motion data sets")

parser.add_argument(
    "--cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/preprocessing/thor.yaml",
    required = False,
    help = "configuration file comprising preprocessing information"
)

args = parser.parse_args()

class PreprocessBenchmark():
    """ Preprocess Benchmark object
    """
    def __init__(self,
                 cfg,
                 ds):
        self.cfg = cfg
        self.save_dir = os.path.join(cfg["preprocessing"]["save_dir"], ds) if ds else cfg["preprocessing"]["save_dir"]
        self.root = os.path.join(cfg["preprocessing"]["root_dir"], ds) if ds else cfg["preprocessing"]["root_dir"]
        self.obs_len = cfg["obs_len"]
        self.sub_traj_len = cfg["obs_len"] + cfg["pred_len"]
        self.modes = ["train", "val", "test"]
        self.scaler = StandardScaler()
        self.scaler_vect_dir = StandardScaler()
        self.scaler_dpolar = StandardScaler()


        self.all_trajs, self.all_trajs_vect_dir, self.all_trajs_dirpolar = {}, {}, {}
        self.ds = ds

    def normalize_data(self):
        # we dont touch in the testing data
        all_trajs_vect_dir = [t[1:, :] for t in self.all_trajs_vect_dir["train"]] # drop first time step
        all_trajs_dirpolar = [t[1:, :] for t in self.all_trajs_dirpolar["train"]] # drop first time step
        trajs = np.concatenate(self.all_trajs["train"]).astype(float)
        trajs_vect_dir = np.concatenate(all_trajs_vect_dir).astype(float)
        trajs_polar_dir = np.concatenate(all_trajs_dirpolar).astype(float)
        trajs_vd = np.concatenate(self.all_trajs_vect_dir["train"]).astype(float)
        trajs_pol= np.concatenate(self.all_trajs_dirpolar["train"]).astype(float)


        data = self.scaler.fit_transform(trajs)
        _ = self.scaler_vect_dir.fit(trajs_vect_dir)
        _ =  self.scaler_dpolar.fit(trajs_polar_dir)
        data_vect_dir = self.scaler_vect_dir.transform(trajs_vd)
        data_polar_dir = self.scaler_dpolar.transform(trajs_pol)

        self.xmean, self.xscale, self.ymean, self.yscale = self.scaler.mean_[0], self.scaler.scale_[0], \
                                                     self.scaler.mean_[1], self.scaler.scale_[1]
        self.xmean_vect_dir, self.xscale_vect_dir, self.ymean_vect_dir, self.yscale_vect_dir = self.scaler_vect_dir.mean_[0], self.scaler_vect_dir.scale_[0], \
                                                                                           self.scaler_vect_dir.mean_[1], self.scaler_vect_dir.scale_[1]
        self.drmean_polar, self.drscale_polar, self.dtmean_polar, self.dtscale_polar = self.scaler_dpolar.mean_[0], self.scaler_dpolar.scale_[0], \
                                                                                   self.scaler_dpolar.mean_[1], self.scaler_dpolar.scale_[1]



        return data, data_vect_dir, data_polar_dir


    def save_data(self, min_ped = None, skip = None):
        modes = ["train_raw", "val_raw", "test_raw",
                 "train_vect_dir", "val_vect_dir", "test_vect_dir",
                 "train_dirpolar", "val_dirpolar", "test_dirpolar",
                 "train_norm",
                 "train_vect_dirnorm",
                 "train_dirpolar_norm"]

        sets = [self.all_trajs["train"], self.all_trajs["val"], self.all_trajs["test"],
                self.all_trajs_vect_dir["train"], self.all_trajs_vect_dir["val"], self.all_trajs_vect_dir["test"],
                self.all_trajs_dirpolar["train"], self.all_trajs_dirpolar["val"], self.all_trajs_dirpolar["test"],
                self.train_set,
                self.train_set_vect_dir,
                self.train_set_dirpolar]

        ps = os.path.join(self.save_dir, f"pp_skip{skip}_minped{min_ped}")
        for i, mode in enumerate(modes):
            if not os.path.exists(ps):
                os.makedirs(ps)
            with open(os.path.join(ps, mode + ".pkl"), "wb") as fp:
                cPickle.dump(sets[i], fp)

        # saving info
        p_info = os.path.join(ps, "info_norm.pkl")
        p_info_vect_dir = os.path.join(ps, "info_vect_dir.pkl")
        p_info_polar_dir = os.path.join(ps, "info_polar_dir.pkl")
        with open(p_info, "wb") as fp:
            cPickle.dump((self.xmean, self.xscale, self.ymean, self.yscale), fp)
        with open(p_info_vect_dir, "wb") as fp:
            cPickle.dump((self.xmean_vect_dir, self.xscale_vect_dir, self.ymean_vect_dir, self.yscale_vect_dir), fp)
        with open(p_info_polar_dir, "wb") as fp:
            cPickle.dump((self.drmean_polar, self.drscale_polar, self.dtmean_polar, self.dtscale_polar), fp)

    def run(self):
        min_ped, skip = self.cfg["preprocessing"]["min_ped"], self.cfg["preprocessing"]["skip"]
        for mode in self.modes:
            p2f = os.path.join(self.root, mode)
            list_files = os.listdir(p2f)
            self.all_trajs[mode], self.all_trajs_vect_dir[mode], self.all_trajs_dirpolar[mode] =  [], [], []
            for f in list_files:
                fp = os.path.join(p2f, f)
                df = pd.read_csv(fp,  delimiter = "\t", names = ["frame_id", "person_id", "x", "y"], index_col = None)
                neigh = df[["person_id", "frame_id"]]
                groups = neigh.groupby("frame_id")
                persons_unique = df["person_id"].unique().tolist()
                for pid in persons_unique:
                    person_data = df[df.person_id == pid]
                    num_subtrajs = int(math.ceil((len(person_data) - self.sub_traj_len + 1) / skip))
                    if num_subtrajs > 0:
                        for idx in range(0, num_subtrajs * skip, skip):
                            frames = person_data.iloc[idx : idx + self.sub_traj_len]["frame_id"].values
                            neigboors = [groups.get_group(fr).person_id.values.tolist() for fr in frames]
                            result = set(neigboors[0])
                            for s in neigboors[1:]: result.intersection_update(s)
                            if len(result) > min_ped:
                                data_xy = person_data.iloc[idx : idx + self.sub_traj_len][["x", "y"]].values.astype(float)
                                data_xy = data_xy
                                if data_xy.shape[0] != self.sub_traj_len: # redundant
                                    continue
                                relative_coords = np.zeros_like(data_xy)
                                relative_coords[1:, :] = data_xy[1:, :] - data_xy[:-1, :]

                                # polar coordinates
                                dr_polar = np.sqrt(relative_coords[:, 0]**2 + relative_coords[:, 1]**2)
                                dtheta_polar = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
                                dpolar_coords = np.stack([dr_polar, dtheta_polar], axis = 1)


                                self.all_trajs[mode].append(data_xy)
                                self.all_trajs_vect_dir[mode].append(relative_coords)
                                self.all_trajs_dirpolar[mode].append(dpolar_coords)

            print(f"{p2f} with {len(self.all_trajs[mode])} samples")
        # normalization
        data, data_vect_dir, data_polar_dir = self.normalize_data()
        train_size = len(self.all_trajs["train"]) *  self.sub_traj_len
        # train set
        self.train_set = [data[i*self.sub_traj_len:(i*self.sub_traj_len + self.sub_traj_len)] for i in range(train_size//  self.sub_traj_len) ]
        self.train_set_vect_dir = [data_vect_dir[i*self.sub_traj_len:(i*self.sub_traj_len + self.sub_traj_len)] for i in range(train_size//self.sub_traj_len)]
        self.train_set_dirpolar = [data_polar_dir[i*self.sub_traj_len:(i*self.sub_traj_len + self.sub_traj_len)] for i in range(train_size//self.sub_traj_len)]
        print("Data preprocessed")
        self.save_data(min_ped, skip)
        print("Data saved")


class PreprocessArgoverse(PreprocessBenchmark):
    """ Preprocess Argoverse object
    """
    def __init__(self, cfg, ds):
        super().__init__(cfg, ds)

    def run(self):
        self.info, dfs = {}, {}
        for mode in self.modes:
            # open file
            p2f = os.path.join(self.root, mode + ".pkl") # list of trajectories
            dfs[mode], self.all_trajs_vect_dir[mode], self.all_trajs[mode], self.all_trajs_dirpolar[mode], self.info[mode] =  [], [], [], [], []
            with open(p2f, "rb") as f:
                dfs[mode] = cPickle.load(f)

            # relative trajectories
            for traj in dfs[mode]:
                info_df = traj[["TIMESTAMP","TRACK_ID","OBJECT_TYPE","CITY_NAME","DATASET_ID"]].copy()
                data_xy = traj[["X", "Y"]].values
                relative_coords = np.zeros_like(data_xy)
                relative_coords[1:, :] = data_xy[1:, :] - data_xy[:-1, :]

                # polar
                dr_polar = np.sqrt(relative_coords[:, 0]**2 + relative_coords[:, 1]**2)
                dtheta_polar = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
                dpolar_coords = np.stack([dr_polar, dtheta_polar], axis = 1)

                if mode == "train":
                    self.all_trajs_vect_dir[mode].append(relative_coords)
                    self.info[mode].append(info_df)
                    self.all_trajs_dirpolar[mode].append(dpolar_coords)

                else: # val or test
                    self.all_trajs_vect_dir[mode].append((relative_coords, info_df))
                    self.all_trajs_dirpolar[mode].append((dpolar_coords, info_df))
                self.all_trajs[mode].append(data_xy)

        data, data_vect_dir, data_polar_dir = self.normalize_data()
        train_size = len(self.all_trajs["train"]) *  self.sub_traj_len

        self.train_set = [(data[i*self.sub_traj_len:(i*self.sub_traj_len + self.sub_traj_len)], \
                           self.info["train"][i]) \
                            for i in range(train_size//self.sub_traj_len)]

        self.train_set_vect_dir = [(data_vect_dir[i*self.sub_traj_len:(i*self.sub_traj_len + self.sub_traj_len)], \
                              self.info["train"][i]) \
                              for i in range(train_size//self.sub_traj_len)]
        self.train_set_dirpolar = [(data_polar_dir[i*self.sub_traj_len:(i*self.sub_traj_len + self.sub_traj_len)], \
                        self.info["train"][i]) \
                        for i in range(train_size//self.sub_traj_len)]
        print("Data preprocessed")
        self.save_data()
        print("Data saved")


class PreprocessThor(PreprocessArgoverse):
    def __init__(self, cfg, ds):
        super().__init__(cfg, ds)

    def run(self):
        self.info, dfs = {}, {}
        for mode in self.modes:
            # open file
            p2f = os.path.join(self.root, mode + ".pkl") # list of trajectories
            dfs[mode], self.all_trajs_vect_dir[mode], self.all_trajs[mode], self.all_trajs_dirpolar[mode], self.info[mode] =  [], [], [], [], []
            with open(p2f, "rb") as f:
                dfs[mode] = cPickle.load(f)
            for traj in dfs[mode]:
                info_df = traj[["Frame","Time","Person_ID","Description"]].copy()
                data_xy = traj[["X", "Y"]].values / 1000 # to m
                relative_coords = np.zeros_like(data_xy)
                relative_coords[1:, :] = data_xy[1:, :] - data_xy[:-1, :]

                # polar
                dr_polar = np.sqrt(relative_coords[:, 0]**2 + relative_coords[:, 1]**2)
                dtheta_polar = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
                dpolar_coords = np.stack([dr_polar, dtheta_polar], axis = 1)

                if mode == "train":
                    self.all_trajs_vect_dir[mode].append(relative_coords)
                    self.info[mode].append(info_df)
                    self.all_trajs_dirpolar[mode].append(dpolar_coords)
                else: # val or test
                    self.all_trajs_vect_dir[mode].append((relative_coords, info_df))
                    self.all_trajs_dirpolar[mode].append((dpolar_coords, info_df))
                self.all_trajs[mode].append(data_xy)

        data, data_vect_dir, data_polar_dir = self.normalize_data()
        train_size = len(self.all_trajs["train"]) *  self.sub_traj_len

        self.train_set = [(data[i*self.sub_traj_len:(i*self.sub_traj_len + self.sub_traj_len)], \
                           self.info["train"][i]) \
                            for i in range(train_size//self.sub_traj_len)]

        self.train_set_vect_dir = [(data_vect_dir[i*self.sub_traj_len:(i*self.sub_traj_len + self.sub_traj_len)], \
                              self.info["train"][i]) \
                              for i in range(train_size//self.sub_traj_len)]

        self.train_set_dirpolar = [(data_polar_dir[i*self.sub_traj_len:(i*self.sub_traj_len + self.sub_traj_len)], \
                        self.info["train"][i]) \
                        for i in range(train_size//self.sub_traj_len)]
        print("Data preprocessed")
        self.save_data()
        print("Data saved")


with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

data = cfg["data"]
print("Processing", data)
if data == "benchmark":
    datasets = cfg["datasets"]
    for ds in datasets:
        pp = PreprocessBenchmark(cfg, ds)
        pp.run()
elif data == "argoverse":
    pp = PreprocessArgoverse(cfg, None)
    pp.run()
elif data == "thor":
    pp = PreprocessThor(cfg, None)
    pp.run()