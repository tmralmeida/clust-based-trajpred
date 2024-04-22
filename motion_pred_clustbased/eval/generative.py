import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import Accuracy
from .common import MeanSquaredError, WeightedADE, WeightedFDE, FDE, Evaluator


class GenEvaluator(Evaluator):
    """Full framework evaluator"""

    def __init__(
        self, dl, cfg, inputs_type, outputs_type, device, writer=None, **kwargs
    ):
        super().__init__(dl, cfg, inputs_type, outputs_type, device, writer=writer)
        self.add_metrics = False
        if self.model == "vae-clust-based":
            model_name = "cvae"
            self.n_preds = cfg["cvae"]["hyperparameters"]["variety"]["n_preds"]
        elif self.model == "gan-clust-based":
            model_name = "cgan"
            self.n_preds = cfg["cgan"]["hyperparameters"]["variety"]["n_preds"]
        elif self.model == "anet":
            model_name = "anet"
            self.n_preds = cfg["anet"]["hyperparameters"]["variety"]["n_preds"]
        else:
            self.n_preds = cfg["hyperparameters"]["variety"]["n_preds"]
        if self.n_preds > 1:
            self.topk_str = f"top{str(self.n_preds)}"
            self.ade = {
                "top1": MeanSquaredError(squared=False).to(device),
                self.topk_str: MeanSquaredError(squared=False).to(device),
                "weighted": WeightedADE().to(device),
            }

            self.fde = {
                "top1": FDE().to(device),
                self.topk_str: FDE().to(device),
                "weighted": WeightedFDE().to(device),
            }

            if (
                self.model in ["vae-clust-based", "gan-clust-based"]
                and cfg[model_name]["condition"]["inference"] != "recognizer"
            ) or self.model == "anet":
                self.add_metrics = True  # additional metrics
                self.ade.update(
                    {
                        "max": MeanSquaredError(squared=False).to(device),
                        "avgtraj": MeanSquaredError(squared=False).to(device),
                    }
                )
                self.fde.update({"max": FDE().to(device), "avgtraj": FDE().to(device)})
                self.accuracy = Accuracy(
                    task="multiclass", num_classes=kwargs["n_clusters"]
                )
        self.path_save = os.path.join(self.save_dir, "plots")

    def __updatetopkmetrics(self, out: dict, ground_truth: torch.Tensor, **kwargs):
        assert self.n_preds > 1, "There is a problem in topk update metrics"
        keys = self.ade.keys()
        for k in keys:
            if k == "weighted":
                self.ade[k].update(
                    preds=out["y_hat_kpreds"],
                    target=ground_truth,
                    weights=out["weights"],
                )
                self.fde[k].update(
                    preds=out["y_hat_kpreds"],
                    target=ground_truth,
                    weights=out["weights"],
                )
            else:
                self.ade[k].update(preds=out[f"y_hat_{k}"], target=ground_truth)
                self.fde[k].update(preds=out[f"y_hat_{k}"], target=ground_truth)
        if self.model not in ["vae-clust-based", "gan-clust-based", "anet"] or not self.add_metrics:
            return
        self.accuracy.update(kwargs["rank_prop_out"], kwargs["soft_cl_label"])

    def __resettopkmetrics(self):
        assert self.n_preds > 1, "There is a problem in topk reset metrics"
        keys = self.ade.keys()
        for k in keys:
            self.ade[k].reset()
            self.fde[k].reset()
        if self.model not in ["vae-clust-based", "gan-clust-based", "anet"] or not self.add_metrics:
            return
        self.accuracy.reset()

    def __computetopkmetrics(self):
        self.ade_res_top1 = self.ade["top1"].compute().item()
        self.ade_res_topk = self.ade[self.topk_str].compute().item()
        self.wade_res = self.ade["weighted"].compute().item()
        self.fde_res_top1 = self.fde["top1"].compute().item()
        self.fde_res_topk = self.fde[self.topk_str].compute().item()
        self.wfde_res = self.fde["weighted"].compute().item()

        self.ade_res, self.fde_res = self.ade_res_top1, self.fde_res_top1
        if self.model in ["vae-clust-based", "gan-clust-based", "anet"] and self.add_metrics:
            self.ade_res_max = self.ade["max"].compute().item()
            self.fde_res_max = self.fde["max"].compute().item()
            self.ade_res_avg = self.ade["avgtraj"].compute().item()
            self.fde_res_avg = self.fde["avgtraj"].compute().item()
            self.acc = self.accuracy.compute().item()

    def __logging_topkmetrics(self, epoch: int, **kwargs):
        print(
            f"---- Epoch %d metrics ----\n \
                   top1_ADE: %1.3f\n \
                   top1_FDE: %1.3f\n\n \
                   {self.topk_str}_ADE: %1.3f\n \
                   {self.topk_str}_FDE: %1.3f\n\n \
                   wADE: %1.3f\n \
                   wFDE: %1.3f"
            % (
                epoch,
                self.ade_res_top1,
                self.fde_res_top1,
                self.ade_res_topk,
                self.fde_res_topk,
                self.wade_res,
                self.wfde_res,
            )
        )
        if self.model in ["vae-clust-based", "gan-clust-based", "anet"] and self.add_metrics:
            print(
                "---- Additional metrics ----\n \
                    max_ADE: %1.3f\n \
                    max_FDE: %1.3f\n \
                    avgtraj_ADE: %1.3f\n \
                    avgtraj_FDE: %1.3f\n"
                % (
                    self.ade_res_max,
                    self.fde_res_max,
                    self.ade_res_avg,
                    self.fde_res_avg,
                )
            )
            print(f"Acc rank proposals: {self.acc:.2f}\n")

    def __write_topkmetrics(self, epoch: int):
        self.writer.add_scalar("top1_ADE", self.ade_res_top1, epoch)
        self.writer.add_scalar("top1_FDE", self.fde_res_top1, epoch)
        self.writer.add_scalar(f"{self.topk_str}_ADE", self.ade_res_topk, epoch)
        self.writer.add_scalar(f"{self.topk_str}_FDE", self.fde_res_topk, epoch)
        if self.model in ["vae-clust-based", "gan-clust-based", "anet"] and self.add_metrics:
            self.writer.add_scalar("max_ADE", self.ade_res_max, epoch)
            self.writer.add_scalar("max_FDE", self.fde_res_max, epoch)
            self.writer.add_scalar("avg_ADE", self.ade_res_avg, epoch)
            self.writer.add_scalar("avg_FDE", self.ade_res_avg, epoch)

    def variety_loss(
        self,
        gen: nn.Module,
        x: torch.Tensor,
        class_preds: torch.Tensor,
        gt_x: torch.Tensor,
        gt_y: torch.Tensor,
        cond_type=None,
    ) -> torch.Tensor:
        mses, predictions, predictions_vels = [], [], []
        new_topk_cond = (
            (
                self.model in ["vae-clust-based", "gan-clust-based"]
                and cond_type
                in ["centroids", "silhouette", "neighbors-ds", "neighbors-fs"]
            )
            or self.model == "anet"
            and cond_type != "recognizer"
        ) and self.n_preds > 1
        if new_topk_cond:
            sorted_probs, idxs = torch.sort(class_preds, dim=-1, descending=True)
            weights = sorted_probs.clone()
        else:
            weights = torch.ones((x.shape[0], self.n_preds)) / self.n_preds

        with torch.no_grad():
            for k in range(
                self.n_preds
                if self.model not in ["vae-clust-based", "gan-clust-based", "anet"]
                else gen.k_value
            ):
                if new_topk_cond:
                    class_preds = idxs[:, k]
                pred = gen(x, labels=class_preds, coord_scaler=self.cs, x_raw=gt_x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                # check if generator or predictor
                work_as_pred = (
                    True if pred.size(1) == gt_y.size(1) else False
                )  # working as a predictor
                if work_as_pred:
                    real_dist_y = self.cs.denorm_increment(gt_x, pred.clone())
                    y = gt_y.clone()
                else:
                    first_ts = gt_x[:, 0, :].unsqueeze(dim=1).clone()
                    out_last = self.cs.denorm_increment(first_ts, pred.clone())
                    real_dist_y = torch.cat([first_ts, out_last], dim=1)
                    y = torch.cat([gt_x, gt_y], dim=1)
                predictions_vels.append(pred)
                predictions.append(real_dist_y)
                loss = F.mse_loss(real_dist_y, y, reduction="none")
                loss = loss.mean(dim=-1).mean(dim=-1)  # keeping batch_dims
                mses.append(loss)
            mses = torch.stack(mses, dim=1)[:, : self.n_preds]
            idx = mses.argmin(dim=1)
            predictions = torch.stack(predictions, dim=0)
            out = []
            for i in range(predictions.size(1)):
                out.append(predictions[idx[i], i, :, :])

            y_hat = (
                torch.stack(out, dim=0)
                if self.n_preds == 1
                else {
                    "y_hat_top1": predictions[0, ...],
                    f"y_hat_{self.topk_str}": torch.stack(out, dim=0),
                    "y_hat_kpreds": predictions,
                    "weights": weights,
                }
            )

            if new_topk_cond:
                predictions_vels = torch.stack(predictions_vels, dim=0)
                sorted_probs = sorted_probs.T.unsqueeze(dim=-1).unsqueeze(
                    dim=-1
                )  # (bs, n_labels) and predictions -> (n_labels, bs, ts, 2)
                avg_trajs = (sorted_probs * predictions_vels).sum(dim=0)
                if work_as_pred:
                    real_dist_y = self.cs.denorm_increment(gt_x, avg_trajs)
                else:
                    out_last = self.cs.denorm_increment(first_ts, avg_trajs.clone())
                    real_dist_y = torch.cat([first_ts, out_last], dim=1)
                y_hat.update({"y_hat_avgtraj": real_dist_y})
        return y, y_hat

    def evaluate(self, model, epoch, clustering=None, cond_state=None, anet=None):
        model.eval()
        true_soft_labels, proc_labels = None, None
        with torch.no_grad():
            with tqdm(self.dl, unit="batch") as tval:
                obss, outs, preds = [], [], []
                proc_labels = None
                right_lbls, n_sples = 0, 0
                for batch in tval:
                    tval.set_description("Eval gen model")
                    if self.ds == "benchmark":
                        (
                            obs,
                            pred,
                            obs_vec_dir,
                            pred_vect_dir,
                            obs_polar,
                            pred_polar,
                        ) = batch
                    else:  # argoverse or thor
                        (
                            obs,
                            pred,
                            obs_vec_dir,
                            pred_vect_dir,
                            obs_polar,
                            pred_polar,
                            raw_labels,
                        ) = batch
                    obs, pred, obs_vec_dir, pred_vect_dir, obs_polar, pred_polar = (
                        obs.to(self.device),
                        pred.to(self.device),
                        obs_vec_dir.to(self.device),
                        pred_vect_dir.to(self.device),
                        obs_polar.to(self.device),
                        pred_polar.to(self.device),
                    )
                    (
                        obs_norm,
                        pred_norm,
                        obs_vectdir_norm,
                        pred_vectdir_norm,
                        obs_polar_norm,
                        pred_polar_norm,
                    ) = self.cs.norm_input(
                        x=obs,
                        y=pred,
                        x_vect_dir=obs_vec_dir,
                        y_vect_dir=pred_vect_dir,
                        x_polar=obs_polar,
                        y_polar=pred_polar,
                    )

                    if self.inputs_type == "dx, dy":
                        inp_obs, inp_pred = obs_vectdir_norm, pred_vectdir_norm
                    elif self.inputs_type == "px, py":
                        inp_obs, inp_pred = obs_polar_norm, pred_polar_norm
                    elif self.inputs_type == "dx, dy, px, py":
                        inp_obs, inp_pred = torch.cat(
                            [obs_vectdir_norm, obs_polar_norm], -1
                        ), torch.cat([pred_vectdir_norm, pred_polar_norm], -1)
                    else:
                        inp_obs, inp_pred = (
                            (obs_vectdir_norm, pred_vectdir_norm)
                            if self.outputs_type == "vect_dir"
                            else (obs_polar_norm, pred_polar_norm)
                        )
                    full_traj = torch.cat([inp_obs, inp_pred], dim=1)
                    if self.model in ["sup-cgan", "sup-cvae"]:
                        proc_labels = raw_labels[:, 0]
                    elif self.model in ["sc-gan", "vae-clust-based", "gan-clust-based"]:
                        proc_labels = cond_state.get_conditions(
                            clustering,
                            inp_obs,
                            "inference",
                            x_raw=obs,
                            y=inp_pred,
                            gen=model,
                        )
                        true_soft_labels = clustering.get_labels(
                            full_traj, None
                        )  # recognizer
                        if cond_state.inf_cond_type in [
                            "centroids",
                            "silhouette",
                            "neighbors-ds",
                            "neighbors-fs",
                        ]:
                            labels2comp = proc_labels.argmax(dim=1)
                            right_lbls += (
                                (labels2comp == true_soft_labels).int().sum().item()
                            )
                            n_sples += len(batch[0])

                    elif self.model == "anet":
                        true_soft_labels = clustering.get_labels(full_traj, None)
                        anet.eval()
                        inp_anet = inp_obs.clone()
                        if anet.inp_seq_len == inp_obs.size(1):
                            out_anet = anet(inp_anet)
                        else:
                            out_anet = anet.get_inference_out(
                                inp_anet, model, true_soft_labels
                            )
                        proc_labels = out_anet.clone()
                        right_lbls += (
                            (out_anet.argmax(dim=-1) == true_soft_labels)
                            .int()
                            .sum()
                            .item()
                        )
                        n_sples += len(batch[0])

                    # results
                    gt, out = self.variety_loss(
                        model,
                        inp_obs,
                        proc_labels,
                        obs,
                        pred,
                        cond_type=cond_state.inf_cond_type
                        if self.model in ["vae-clust-based", "gan-clust-based"]
                        else None,
                    )

                    # get maximum metrics
                    if (
                        (
                            self.model in ["vae-clust-based", "gan-clust-based"]
                            and cond_state.inf_cond_type
                            in [
                                "centroids",
                                "silhouette",
                                "neighbors-ds",
                                "neighbors-fs",
                            ]
                        )
                        or self.model == "anet"
                    ) and self.n_preds > 1:
                        _, out_recog = self.variety_loss(
                            model,
                            inp_obs,
                            true_soft_labels,
                            obs,
                            pred,
                            cond_type="recognizer",
                        )
                        out.update({"y_hat_max": out_recog["y_hat_top1"]})
                    if self.n_preds == 1:
                        self.update_metrics(out, gt)
                    else:
                        self.__updatetopkmetrics(
                            out,
                            gt,
                            soft_cl_label=true_soft_labels,
                            rank_prop_out=proc_labels,
                        )
                    obss.append(obs)
                    outs.append(out)
                    preds.append(gt)
                if self.save_plot_trajs:
                    self.save_plot(obss, outs, preds, epoch)
                if self.n_preds > 1:
                    self.__computetopkmetrics()  # compute metrics:
                    self.__logging_topkmetrics(epoch)  # log metrics
                    self.__resettopkmetrics()  # reset metrics
                    if self.writer is not None:
                        self.__write_topkmetrics(epoch)
                else:
                    self.compute_metrics()
                    self.logging_metrics(epoch)
                    self.reset_metrics()
                    if self.writer is not None:
                        self.write_metrics(epoch)
