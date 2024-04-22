import yaml
import os
import _pickle as cPickle
import torch

from ..clustering.feature_space import clustering_dict


def load_config(path, default_path):
    """Loads config file.
    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    with open(path, "r") as f:
        cfg_spec = yaml.safe_load(f)

    inherit_from = cfg_spec.get("inherit_from")

    if inherit_from:
        cfg = load_config(inherit_from, default_path)
    else:
        with open(default_path, "r") as f:
            cfg = yaml.safe_load(f)

    update_recursive(cfg, cfg_spec)
    return cfg


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    """
    for k, v in dict2.items():
        # Add item if not yet in dict1
        if k not in dict1:
            dict1[k] = None
        # Update
        if isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def concat_subtrajs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Partial trajectories concatenation
    Args:
        x (torch.Tensor): observed trajectory
        y (torch.Tensor): predicted/gt trajectory
    Returns:
        torch.Tensor: full trajectory concatenated
    """
    out = torch.cat([x, y], dim=1)
    assert out.shape[1] == x.shape[1] + y.shape[1]
    return out


class CoordsScaler(object):
    """Handler for coordinate scaler"""

    def __init__(self, cfg) -> None:
        self.ds = cfg["data"]["dataset"]
        self.output = cfg["data"]["output"]
        data_dir = cfg["data"]["data_dir"]
        ds_target = str(cfg["data"]["dataset_target"])
        fns = f"pp_skip{str(cfg['data']['skip'])}_minped{str(cfg['data']['min_ped'])}"
        info_path_polar = os.path.join(data_dir, ds_target, fns, "info_polar_dir.pkl")
        info_path_rel = os.path.join(data_dir, ds_target, fns, "info_vect_dir.pkl")
        info_path_raw = os.path.join(data_dir, ds_target, fns, "info_norm.pkl")
        with open(info_path_polar, "rb") as f:
            self.rmean, self.rscale, self.tmean, self.tscale = cPickle.load(f)
        with open(info_path_rel, "rb") as f:
            (
                self.xmean_rel,
                self.xscale_rel,
                self.ymean_rel,
                self.yscale_rel,
            ) = cPickle.load(f)
        with open(info_path_raw, "rb") as f:
            (
                self.xmean_raw,
                self.xscale_raw,
                self.ymean_raw,
                self.yscale_raw,
            ) = cPickle.load(f)
        if self.output == "vect_dir":  # relative outputs
            self.xmean_main, self.xscale_main, self.ymean_main, self.yscale_main = (
                self.xmean_rel,
                self.xscale_rel,
                self.ymean_rel,
                self.yscale_rel,
            )
        elif self.output == "polar":
            self.xmean_main, self.xscale_main, self.ymean_main, self.yscale_main = (
                self.rmean,
                self.rscale,
                self.tmean,
                self.tscale,
            )
        else:
            raise NotImplementedError(f"cfg['data']['output']")

    def norm_coords(self, x, mode="raw"):
        if mode == "raw":
            xmean, xscale, ymean, yscale = (
                self.xmean_raw,
                self.xscale_raw,
                self.ymean_raw,
                self.yscale_raw,
            )
        elif mode == "vect_dir":
            xmean, xscale, ymean, yscale = (
                self.xmean_rel,
                self.xscale_rel,
                self.ymean_rel,
                self.yscale_rel,
            )
        elif mode == "polar":
            xmean, xscale, ymean, yscale = (
                self.rmean,
                self.rscale,
                self.tmean,
                self.tscale,
            )
        else:
            raise NotImplementedError(mode)
        _x, y = x[:, :, 0].unsqueeze(dim=-1), x[:, :, 1].unsqueeze(dim=-1)
        xz = (_x - xmean) / xscale
        xy = (y - ymean) / yscale
        normed_pt = torch.cat([xz, xy], dim=-1)
        return normed_pt

    def denorm_coords(self, pred):
        pred[:, :, 0] = (self.xscale_main * pred[:, :, 0]) + self.xmean_main
        pred[:, :, 1] = (self.yscale_main * pred[:, :, 1]) + self.ymean_main
        if self.output == "polar":
            pred[:, :, 0] = pred[:, :, 0].clone() * torch.cos(
                pred[:, :, 1].clone()
            )  # dx = rcos\theta
            pred[:, :, 1] = pred[:, :, 0].clone() * torch.sin(
                pred[:, :, 1].clone()
            )  # dy = rsin\theta
        return pred

    def increment_coords(self, x, y):
        lp = x[:, -1, :]
        y[:, 0, :] += lp
        return torch.cumsum(y, dim=1)

    def norm_input(
        self,
        x=None,
        y=None,
        x_vect_dir=None,
        y_vect_dir=None,
        x_polar=None,
        y_polar=None,
    ):
        norm_inps = []
        if x is not None and y is not None:
            for inp in (x, y):
                norm_inps.append(self.norm_coords(inp, mode="raw"))
        if x_vect_dir is not None and y_vect_dir is not None:
            for inp in (x_vect_dir, y_vect_dir):
                norm_inps.append(self.norm_coords(inp, mode="vect_dir"))
        if x_polar is not None and y_polar is not None:
            for inp in (x_polar, y_polar):
                norm_inps.append(self.norm_coords(inp, mode="polar"))
        return norm_inps

    def denorm_increment(self, x, y):
        last_y = self.denorm_coords(y)
        if self.output == "vect_dir" or self.output == "polar":
            last_y = self.increment_coords(x, last_y)
        return last_y


def get_nsamples(data_loader, cfg, inputs, coord_scaler):
    """retrieve n samples to be clustered
    Args:
        data_loader (): torch dataloader
        cfg (dict): config file
        inputs(str): inputs_type
    Returns:
        out, lbl: samples from the training set, respective sup label list
    """
    N = cfg["clustering"]["nsamples"]
    data_cfg = cfg["data"]
    output = data_cfg["output"]
    ds = data_cfg["dataset"]
    obs_len = data_cfg["obs_len"]
    pred_len = data_cfg["pred_len"]
    x, y, lbl, n = [], [], [], 0
    for batch in data_loader:
        if ds == "benchmark":
            obs, pred, obs_vect_dir, pred_vect_dir, obs_polar, pred_polar = batch
        elif ds == "argoverse" or ds == "thor":
            (
                obs,
                pred,
                obs_vect_dir,
                pred_vect_dir,
                obs_polar,
                pred_polar,
                raw_labels,
            ) = batch

        obs_vect_dir, pred_vect_dir, obs_polar, pred_polar = (
            coord_scaler.norm_coords(obs_vect_dir, mode="vect_dir"),
            coord_scaler.norm_coords(pred_vect_dir, mode="vect_dir"),
            coord_scaler.norm_coords(obs_polar, mode="polar"),
            coord_scaler.norm_coords(pred_polar, mode="polar"),
        )

        if inputs == "dx, dy":
            inp_obs, inp_pred = obs_vect_dir, pred_vect_dir
        elif inputs == "px, py":
            inp_obs, inp_pred = obs_polar, pred_polar
        elif inputs == "dx, dy, px, py":
            inp_obs, inp_pred = torch.cat([obs_vect_dir, obs_polar], -1), torch.cat(
                [pred_vect_dir, pred_polar], -1
            )
        else:  # based on full trajectory
            inp_obs, inp_pred = (
                (obs_vect_dir, pred_vect_dir)
                if output == "vect_dir"
                else (obs_polar, pred_polar)
            )

        x.append(inp_obs)
        y.append(inp_pred)
        if ds == "argoverse" or ds == "thor":
            lbl.append(raw_labels)
        n += inp_obs.size(0)
        if n > N:
            break
    x = torch.cat(x, dim=0)[:N]
    y = torch.cat(y, dim=0)[:N]
    if ds == "argoverse" or ds == "thor":
        lbl = torch.cat(lbl, dim=0)[:N]
    out = torch.cat([x, y], dim=1)
    assert (
        out.shape[1] == obs_len - 1 + pred_len
    ), "Something wrong with the input shapes"
    return out, lbl


def get_clustering(cl_name):
    return clustering_dict[cl_name]


def get_batch(
    batch,
    coord_scaler,
    model_type,
    dataset_name,
    inputs_type,
    device,
    clustering=None,
    **kwargs,
):
    full_traj, proc_labels = None, None
    if dataset_name == "benchmark":
        obs, pred, obs_vect_dir, pred_vect_dir, obs_polar, pred_polar = batch
    elif dataset_name == "argoverse" or dataset_name == "thor":
        (
            obs,
            pred,
            obs_vect_dir,
            pred_vect_dir,
            obs_polar,
            pred_polar,
            raw_labels,
        ) = batch

    if inputs_type == "dx, dy":
        inp_obs, inp_pred = coord_scaler.norm_coords(
            obs_vect_dir, mode="vect_dir"
        ), coord_scaler.norm_coords(pred_vect_dir, mode="vect_dir")
    elif inputs_type == "px, py":
        inp_obs, inp_pred = coord_scaler.norm_coords(
            obs_polar, mode="polar"
        ), coord_scaler.norm_coords(pred_polar, mode="polar")
    elif inputs_type == "dx, dy, px, py":
        obs_vect_dir, pred_vect_dir = coord_scaler.norm_coords(
            obs_vect_dir, mode="vect_dir"
        ), coord_scaler.norm_coords(pred_vect_dir, mode="vect_dir")
        obs_polar, pred_polar = coord_scaler.norm_coords(
            obs_polar, mode="polar"
        ), coord_scaler.norm_coords(pred_polar, mode="polar")
        inp_obs, inp_pred = torch.cat([obs_vect_dir, obs_polar], -1).to(
            device
        ), torch.cat([pred_vect_dir, pred_polar], -1).to(device)
    else:
        inp_obs, inp_pred = (
            (obs_vect_dir, pred_vect_dir)
            if kwargs["output_type"] == "vect_dir"
            else (obs_polar, pred_polar)
        )
    if model_type in ["sup-cgan", "sup-cvae"]:
        assert dataset_name in [
            "argoverse",
            "thor",
        ], f"{dataset_name} does not provide labels"
        proc_labels = raw_labels[:, 0].to(device)
    elif model_type in ["sc-gan", "vae-clust-based", "gan-clust-based", "anet"]:
        full_traj = concat_subtrajs(inp_obs, inp_pred).to(device)
        proc_labels = clustering.get_labels(full_traj, None).to(device)
    return obs.to(device), pred.to(device), inp_obs, inp_pred, full_traj, proc_labels


def cat_class_emb(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() != x.dim():
        one_hot = labels.unsqueeze(dim=1)
        class_emb_inp = one_hot.repeat(1, x.size(1), 1)
    else:
        class_emb_inp = labels
    return torch.cat([x, class_emb_inp], dim=-1)


def merge_dicts(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            kd = dict_1[key]
            if not isinstance(value, list):
                new_val = [value]
            if not isinstance(dict_1[key], list):
                kd = [dict_1[key]]
            new_val.extend(kd)
            dict_3[key] = new_val
    return dict_3
