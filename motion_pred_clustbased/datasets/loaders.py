import os
import torch
from torch.utils.data import Dataset
import _pickle as cPickle
from ..utils.common import *


def load_ds(mode : str, cfg : dict, clusterer = None):
    """ Loading dataset according to the mode (train, val, or test) and the config file
    Args:
        mode (str): `train`, `val`, or `test`
        cfg (dict): config file
    Raises:
        NameError: if the dataset name is not accepted
    Returns:
        [type]: dataset object
    """
    ds = cfg["dataset"]
    tested_ds = str(cfg["dataset_target"])

    root = os.path.join(cfg["data_dir"], tested_ds, f"pp_skip{str(cfg['skip'])}_minped{str(cfg['min_ped'])}")
    obs_len = cfg["obs_len"]
    if ds == "sCREEN":
        file = mode + ".pkl"
        root = os.path.join(os.path.join(cfg["data_dir"], tested_ds), file)

    elif ds == "benchmark":
        return Benchmark(
            root = root,
            mode = mode,
            obs_len = obs_len
        )
    elif ds == "argoverse":
        return Argoverse(
            root = root,
            mode = mode,
            obs_len = obs_len
        )
    elif ds == "thor":
        return Thor(
            root = root,
            mode = mode,
            obs_len = obs_len

        )
    else:
        raise NameError(f"{ds} not available.")


class Benchmark(Dataset):
    """ Loader for benchmark dataset
    """
    def __init__(self,
                 root,
                 mode = "train",
                 obs_len = 8,
                 ):
        self.root = root
        self.mode = mode
        self.obs_len = obs_len

        # open files
        self.files = ["_raw", "_vect_dir", "_dirpolar", "_vect_dirnorm"] if self.mode == "train" else ["_raw", "_vect_dir", "_dirpolar"]
        self.data = {}
        for _f in self.files:
            fp = os.path.join(root, mode + _f + ".pkl")
            with open(fp, "rb") as f:
                self.data[_f] = cPickle.load(f)


    def __getitem__(self, index):
        x, y = self.data[self.files[0]][index][:self.obs_len], self.data[self.files[0]][index][self.obs_len:]
        x_vect_dir, y_vect_dir = self.data[self.files[1]][index][1:self.obs_len], self.data[self.files[1]][index][self.obs_len:]
        x_polar, y_polar = self.data[self.files[2]][index][1:self.obs_len], self.data[self.files[2]][index][self.obs_len:]
        return (torch.from_numpy(x).type(torch.float),
                torch.from_numpy(y).type(torch.float),
                torch.from_numpy(x_vect_dir).type(torch.float),
                torch.from_numpy(y_vect_dir).type(torch.float),
                torch.from_numpy(x_polar).type(torch.float),
                torch.from_numpy(y_polar).type(torch.float))

    def __len__(self):
        return len(self.data["_raw"])


class Argoverse(Benchmark):
    """Loader for Argoverse dataset
    """
    def __init__(self,
                 root,
                 mode = "train",
                 obs_len = 8
                 ):
        super().__init__(root, mode = mode, obs_len = obs_len)
        self.labels = ["OTHERS", "AV", "AGENT"]


    def __getitem__(self, index):
        x, y = self.data[self.files[0]][index][:self.obs_len], self.data[self.files[0]][index][self.obs_len:]
        fn_dir = self.data[self.files[1]][index] if self.mode == "train" else self.data[self.files[1]][index][0]
        fn_polar = self.data[self.files[2]][index] if self.mode == "train" else self.data[self.files[2]][index][0]
        x_vect_dir, y_vect_dir = fn_dir[1:self.obs_len], fn_dir[self.obs_len:]
        x_polar, y_polar = fn_polar[1:self.obs_len], fn_polar[self.obs_len:]
        info = self.data[self.files[-1]][index][1]
        sup_label = torch.full((x.shape[0],), self.labels.index(info.iloc[0].OBJECT_TYPE))
        return (torch.from_numpy(x).type(torch.float),
                torch.from_numpy(y).type(torch.float),
                torch.from_numpy(x_vect_dir).type(torch.float),
                torch.from_numpy(y_vect_dir).type(torch.float),
                torch.from_numpy(x_polar).type(torch.float),
                torch.from_numpy(y_polar).type(torch.float),
                sup_label)


class Thor(Argoverse):
    def __init__(self, root, mode="train", obs_len=8):
        super().__init__(root, mode, obs_len)
        self.labels = ["WORKER", "VISITOR", "INSPECTOR"]

    def __getitem__(self, index):
        x, y = self.data[self.files[0]][index][:self.obs_len], self.data[self.files[0]][index][self.obs_len:]
        fn_dir = self.data[self.files[1]][index] if self.mode == "train" else self.data[self.files[1]][index][0]
        fn_polar = self.data[self.files[2]][index] if self.mode == "train" else self.data[self.files[2]][index][0]
        x_vect_dir, y_vect_dir = fn_dir[1:self.obs_len], fn_dir[self.obs_len:]
        x_polar, y_polar = fn_polar[1:self.obs_len], fn_polar[self.obs_len:]
        info = self.data[self.files[-1]][index][1]
        sup_label = torch.full((x.shape[0],), self.labels.index(info.iloc[0].Description)) # supervised label
        return (torch.from_numpy(x).type(torch.float),
                torch.from_numpy(y).type(torch.float),
                torch.from_numpy(x_vect_dir).type(torch.float),
                torch.from_numpy(y_vect_dir).type(torch.float),
                torch.from_numpy(x_polar).type(torch.float),
                torch.from_numpy(y_polar).type(torch.float),
                sup_label)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from tqdm import tqdm

    def run_set(data_iter, data):
        for i in tqdm(range(len(data))):
            if dataset == "benchmark":
                x, y, x_vd, y_vd, x_p, y_p = next(data_iter)
            elif dataset in ["thor", "argoverse"]:
                x, y, x_vd, y_vd, x_p, y_p, label = next(data_iter)
                print("label shape", label.shape)
            print(x.shape, y.shape, x_vd.shape, y_vd.shape, x_p.shape, y_p.shape)

    parser = ArgumentParser(description = "Test SCGAN")

    parser.add_argument(
        "--cfg",
        type = str,
        default = "motion_pred_clustbased/cfg/clustering/feature_space/benchmark.yaml",
        required = False,
        help = "configuration file comprising: networks design choices, hyperparameters, etc."
    )

    args = parser.parse_args()
    cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/clustering/feature_space/default.yaml")
    data_cfg = cfg["data"]
    dataset = data_cfg["dataset"]
    train_ds = load_ds("train", data_cfg)
    print("Train dataset loaded!")
    val_ds = load_ds("val", data_cfg)
    print("Val dataset loaded!")
    test_ds = load_ds("test", data_cfg)
    print("Test dataset loaded!")

    train_ds_iter = iter(train_ds)
    run_set(train_ds_iter, train_ds)
    print("Train set verified!")
    val_ds_iter = iter(val_ds)
    run_set(val_ds_iter, val_ds)
    print("Val set verified!")
    test_ds_iter = iter(test_ds)
    run_set(test_ds_iter, test_ds)
    print("Test set verified!")