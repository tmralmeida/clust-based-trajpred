import torch
from tqdm import tqdm
from torchmetrics import Metric, MeanSquaredError
import os
import matplotlib.pyplot as plt
import numpy as np
from ..utils.common import CoordsScaler


class WeightedADE(Metric):
    """Weighted average displacement error"""

    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        assert (weights.sum(dim=1) >= 0.99).all(), "Weights must sum up to 1"
        assert weights.size(1) == preds.size(
            0
        ), "Weights ndims must match number of preds"
        ades = [torch.pow(pred - target, 2).mean(dim=-1).mean(dim=-1) for pred in preds]
        ades = torch.stack(ades, dim=1)
        loss = (ades * weights).sum(dim=-1)
        self.dist += loss.sum()
        self.total += target.shape[0]

    def compute(self):
        return torch.sqrt(self.dist.float() / self.total)


class FDE(Metric):
    """FDE metric for torchmetrics API"""

    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, "Problem on FDE shapes"
        last_gt, last = target[:, -1, :], preds[:, -1, :]
        loss = torch.norm(last_gt - last, 2, 1)
        self.dist += loss.sum()
        self.total += target.shape[0]  # evaluating bs

    def compute(self):
        return self.dist.float() / self.total


class WeightedFDE(Metric):
    """Weighted average displacement error"""

    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        assert (weights.sum(dim=1) >= 0.99).all(), "Weights must sum up to 1"
        assert weights.size(1) == preds.size(
            0
        ), "Weights ndims must match number of preds"
        fdes = []
        for n_pred in range(preds.size(0)):
            last = preds[n_pred, :, -1, :]
            last_gt = target[:, -1, :]
            fdes.append(torch.norm(last_gt - last, 2, 1))
        fdes = torch.stack(fdes, dim=1)
        loss = (fdes * weights).sum(dim=-1)
        self.dist += loss.sum()
        self.total += target.shape[0]

    def compute(self):
        return self.dist.float() / self.total


class Evaluator:
    """Naive evaluator object"""

    def __init__(
        self, dl, cfg, inputs_type, outputs_type, device, writer=None, **kwargs
    ):
        self.dl = dl
        self.writer = writer
        self.device = device
        self.test_seq_len = cfg["data"]["pred_len"]
        self.ade = MeanSquaredError(squared=False).to(device)
        self.fde = FDE().to(device)
        self.ds = cfg["data"]["dataset"]
        self.cs = CoordsScaler(cfg)
        self.save_plot_trajs = cfg["save"]["plots"]
        self.save_dir = cfg["save"]["path"]
        self.ds_target = cfg["data"]["dataset_target"]
        self.model = cfg["model"]
        self.inputs_type = inputs_type
        self.outputs_type = outputs_type

    def update_metrics(self, out, gt):
        self.ade.update(preds=out, target=gt)
        self.fde.update(preds=out, target=gt)

    def reset_metrics(self):
        self.ade.reset()
        self.fde.reset()

    def compute_metrics(self):
        self.ade_res = self.ade.compute().item()
        self.fde_res = self.fde.compute().item()

    def logging_metrics(self, epoch: int, **kwargs):
        print(
            "---- Epoch %d metrics ----\n \
                   ADE: %1.3f\n \
                   FDE: %1.3f\n"
            % (epoch, self.ade_res, self.fde_res)
        )
        if "acc" in kwargs.keys() and kwargs["acc"] is not None:  # acc
            print("inf_ACC: % 1.3f\n" % (kwargs["acc"]))

    def write_metrics(self, epoch):
        self.writer.add_scalar("ADE", self.ade_res, epoch)
        self.writer.add_scalar("FDE", self.fde_res, epoch)

    def save_plot(self, xs, ys, gts, epoch):
        fig = plt.figure(figsize=(15, 15))
        gen_mode = self.model in ["sc-gan", "ft-gan"]
        plt.title(
            "Val/Test set input features alignment"
            if gen_mode
            else "Val/Test set trajectories...",
            fontsize=29,
        )
        for i in range(len(xs)):
            x_inp = xs[i].cpu().numpy()
            gt = gts[i].cpu().numpy()
            if (
                (self.model == "sc-gan" and self.n_preds == 1)
                or self.model == "lstm"
                or (self.model in ["van-gan", "ft-gan"] and self.n_preds == 1)
            ):
                y_pred = ys[i].cpu().numpy()
            else:
                y_pred = ys[i]["y_hat_top1"].cpu().numpy()
            for j in range(x_inp.shape[0]):
                x, y, real = x_inp[j, :, :], y_pred[j, :, :], gt[j, :, :]
                if y_pred.shape[1] == self.test_seq_len:
                    plt.plot(
                        x[:, 0],
                        x[:, 1],
                        "b",
                        label="Observed" if i == 0 and j == 0 else "",
                    )  # obs
                    plt.plot(
                        y[:, 0],
                        y[:, 1],
                        "r",
                        label="Prediction" if i == 0 and j == 0 else "",
                    )  # pred
                    plt.plot(
                        real[:, 0],
                        real[:, 1],
                        "g",
                        label="Ground-Truth" if i == 0 and j == 0 else "",
                    )  # gt
                elif y_pred.shape[1] == self.test_seq_len + x_inp.shape[1]:
                    gt_relative_coords = np.zeros_like(real)
                    gt_relative_coords[1:, :] = real[1:, :] - real[:-1, :]
                    ours_relative_coords = np.zeros_like(y)
                    ours_relative_coords[1:, :] = y[1:, :] - y[:-1, :]

                    plt.scatter(
                        ours_relative_coords[:, 0],
                        ours_relative_coords[:, 1],
                        s=5,
                        c="r",
                        label="Generated Displacements" if i == 0 and j == 0 else "",
                    )  # pred
                    plt.scatter(
                        gt_relative_coords[:, 0],
                        gt_relative_coords[:, 1],
                        s=5,
                        c="g",
                        label="Ground-Truth Displacements" if i == 0 and j == 0 else "",
                    )  # pred

        plt.legend(prop={"size": 29}, markerscale=10)
        # plt.axis('off');
        print("saving plot")
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)
        fig.savefig(
            os.path.join(
                self.path_save, f"{self.ds}_{self.ds_target}_{self.model}_ep{epoch}.svg"
            ),
            bbox_inches="tight",
        )
        plt.plot()
        plt.close(fig)

    def evaluate(self, model, epoch, clusterer=None):
        gt_class = None
        model.eval()
        with torch.no_grad():
            with tqdm(self.dl, unit="batch") as tval:
                for batch in tval:
                    tval.set_description("Eval SCGAN")
                    if self.ds == "benchmark" or self.ds == "sCREEN":
                        x, y, x_rel, y_rel = batch
                    else:  # argoverse or thor
                        x, y, x_rel, y_rel, gt_class = batch
                    x, y, x_rel, y_rel = (
                        x.to(self.device),
                        y.to(self.device),
                        x_rel.to(self.device),
                        y_rel.to(self.device),
                    )
                    obs_norm, pred_norm, obs_relnorm, pred_relnorm = self.cs.norm_input(
                        x, y, x_rel, y_rel
                    )

                    if self.inputs_type == "x, y":
                        inp_obs, inp_pred = obs_norm, pred_norm
                    elif self.inputs_type == "dx, dy":
                        inp_obs, inp_pred = obs_relnorm, pred_relnorm
                    elif self.inputs_type == "x, y, dx, dy":
                        inp_obs, inp_pred = torch.cat(
                            [obs_norm, obs_relnorm], -1
                        ), torch.cat([pred_norm, pred_relnorm], -1)

                    if clusterer:
                        gt_class = clusterer.get_labels(inp_obs, inp_pred, None)
                    out = model(inp_obs, clusters_class=gt_class)
                    pred = out.view(out.shape[0], self.test_seq_len, -1)  # (bs, sl, 2)
                    if self.global_norm:
                        pred = self.cs.denorm_coords(pred)
                    if self.relative_inp:
                        pred = self.cs.increment_coords(x, pred)
                    self.update_metrics(pred, y)
                self.compute_metrics()
                self.logging_metrics(epoch)
                self.reset_metrics()
                if self.writer is not None:
                    self.write_metrics(epoch)
