import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..datasets.loaders import *
from .deterministic import Trainer
from .gan_based import GanTrainer
from ..networks.generators import cVAETraj
from ..utils.generative import create_dirs_vae, ConditionState, train_vae
from ..utils.deterministic import (
    save_checkpoints_det as save_cehckpoints_vae,
    save_objects_det as save_objects_vae,
)
from ..clustering.utils import save_clustering_object
from ..utils.common import *
from ..eval.generative import GenEvaluator
from ..clustering.raw_data.clusterers import *


class VAEClusTrainer(Trainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.clusterer_info = cfg["clustering"]
        self.clustering_inputs = cfg["clustering"]["inputs"]

    def get_ds_np(self):
        ds = self.data_cfg["dataset"]
        print(f"\nClustering {ds}...")

        skip, min_ped = self.data_cfg["skip"], self.data_cfg["min_ped"]
        target_dataset = str(self.data_cfg["dataset_target"]) + (
            f"/pp_skip{skip}_minped{min_ped}"
            if ds in ["thor", "argoverse"]
            else f"/pp_skip{skip}_minped{min_ped}"
        )

        dir = os.path.join(self.data_cfg["data_dir"], target_dataset)
        list_files = os.listdir(dir)
        data = {}
        for file in list_files:
            fp = os.path.join(dir, file)
            with open(fp, "rb") as f:
                data[file] = cPickle.load(f)

        vectdir_norm_traj = np.stack(
            data["train_vect_dirnorm.pkl"]
            if ds == "benchmark"
            else [
                data["train_vect_dirnorm.pkl"][i][0]
                for i in range(len(data["train_vect_dirnorm.pkl"]))
            ]
        )
        dirpolar_norm_traj = np.stack(
            data["train_dirpolar_norm.pkl"]
            if ds == "benchmark"
            else [
                data["train_dirpolar_norm.pkl"][i][0]
                for i in range(len(data["train_dirpolar_norm.pkl"]))
            ]
        )

        print("Dataset loaded!")

        if self.clustering_inputs == "vect_dir":
            inputs = vectdir_norm_traj[:, 1:, :]
        if self.clustering_inputs == "polar":
            inputs = dirpolar_norm_traj[:, 1:, :]
        return inputs

    def get_clusterer(self, dataset, n_clusters, out_path=None, writer=None) -> None:
        algorithm = self.clusterer_info["algorithm"]
        clust_cfg = deepcopy(self.clusterer_info[algorithm])
        if algorithm in ["k-means", "k-shape"]:
            clust_cfg["n_clusters"] = n_clusters
            clust = CustomKClust(algorithm, clust_cfg, writer)
            metrics, clusterer = clust.run(dataset)
            save_clustering_object(out_path, clusterer, "0")
        elif algorithm == "hdbscan":
            clust_cfg["n_clusters"] = n_clusters
            input_type = "polar" if self.input_data["polar"] else "vect_dir"
            clust = CustomHDBSCAN(algorithm, clust_cfg, out_path, input_type, writer)
            metrics, _, clusterer = clust.run(dataset)
        elif algorithm == "kmeans_selfcondgan":
            cfg = deepcopy(self.cfg)
            cfg["model"] = "sc-gan"
            cfg_cl = clust_cfg
            cfg_cl["clustering"]["kwargs"]["n_clusters"] = n_clusters
            cfg_cl["clustering"].update({"name": algorithm})
            cfg.update(cfg_cl)
            cfg["data"]["inputs"]["vect_dir"] = False
            cfg["data"]["inputs"]["polar"] = False
            gt = GanTrainer(cfg)
            metrics, clusterer, _ = gt.train()

        elif algorithm == "dbscan":
            raise NotImplementedError(algorithm)
        return metrics, clusterer

    def train(self, *args, **kwargs):
        torch.random.seed()
        # 1st step get clusterer object
        assert self.model_type == "vae-clust-based", "Wrong model"
        print("\nGetting clustering...")
        cvae_hyp = self.cfg["cvae"]["hyperparameters"]
        dataset_np = self.get_ds_np()
        runs_dir, output_dir = create_dirs_vae(
            self.cfg, self.cfg["cvae"]["network"], cvae_hyp, self.inputs
        )
        writer = SummaryWriter(runs_dir)
        if "aug_dataset" in kwargs:
            train_dl = kwargs["aug_dataset"]
            print(
                "\n\n=======================================================Training new model on synthetic dataset=======================================================\n\n"
            )
        else:
            train_ds = load_ds("train", self.data_cfg)
            train_dl = DataLoader(train_ds, self.hyp_cfg["bs"], shuffle=True)

        clust_algo = self.clusterer_info["algorithm"]
        ncl = (
            self.clusterer_info[clust_algo]["n_clusters"]
            if clust_algo in ["k-means", "k-shape"]
            else self.clusterer_info[clust_algo]["clustering"]["kwargs"]["n_clusters"]
        )
        ncl = ncl if len(args) == 0 else args[0]
        metrics_clusterer, clusterer = self.get_clusterer(
            dataset_np, ncl, out_path=output_dir, writer=writer
        )
        self.cfg["cvae"]["network"]["n_labels"] = clusterer.k

        self.common_settings()

        # cVAE model
        cvae = cVAETraj(self.cfg["cvae"]["network"], self.inp_dim, self.device)

        # conditioning handler
        cond_state = ConditionState(
            self.model_type,
            self.cs,
            clusterer,
            self.cfg["cvae"]["condition"],
        )

        # optimizer and scheduler
        opt = optim.Adam(cvae.parameters(), lr=float(cvae_hyp["lr"]), weight_decay=1e-4)
        lr_sched = optim.lr_scheduler.StepLR(opt, int(cvae_hyp["step_size"]))

        # validation and testing objects
        validator = GenEvaluator(
            self.val_dl,
            self.cfg,
            self.inputs,
            self.outputs,
            self.device,
            writer=writer,
            n_clusters=ncl,
        )

        testing = False
        if self.data_cfg["test"]:
            test_ds = load_ds("test", self.data_cfg)
            testing = True
            test_dl = DataLoader(test_ds, cvae_hyp["bs"])
            tester = GenEvaluator(
                test_dl, self.cfg, self.inputs, self.outputs, self.device, n_clusters=ncl
            )

        # metrics
        metrics = {"ade": [], "fde": []}
        metrics.update(
            {
                "sse": [metrics_clusterer["sse"]] * cvae_hyp["max_epochs"],
                "clust_score": [metrics_clusterer["clust_score"]]
                * cvae_hyp["max_epochs"],
            }
        )
        up_topk = isinstance(validator.ade, dict)  # topk metrics
        if up_topk:
            metrics.update(
                {
                    "ade_topk": [],
                    "fde_topk": [],
                    "ade_avg": [],
                    "fde_avg": [],
                }
            )

        cvaes = []
        print("===================================" * 5)
        print("\nTraining cVAE...\n\n")
        for epoch in range(cvae_hyp["max_epochs"]):
            losses = []
            with tqdm(train_dl, unit="batch") as tepoch:
                for batch in tepoch:
                    obs, pred, inp_obs, inp_pred, _, proc_label = get_batch(
                        batch,
                        self.cs,
                        self.model_type,
                        self.ds,
                        self.inputs,
                        self.device,
                        clustering=clusterer,
                        gen=None,
                        cond_state=self.cs,
                        output_type=self.data_cfg["output"],
                    )
                    tepoch.set_description(f"Epoch {epoch + 1}")

                    inp_obs, inp_pred, obs, pred, proc_label = (
                        inp_obs.to(self.device),
                        inp_pred.to(self.device),
                        obs.to(self.device),
                        pred.to(self.device),
                        proc_label.to(self.device),
                    )

                    loss = train_vae(
                        cvae,
                        obs,
                        pred,
                        inp_obs,
                        inp_pred,
                        opt,
                        self.cs,
                        cvae_hyp,
                        labels=proc_label,
                        device=self.device,
                    )

                    losses.append(loss)
                    avg_loss = sum(losses) / len(losses)
                    tepoch.set_postfix({"loss": avg_loss})
                writer.add_scalar("loss_avg", avg_loss, epoch)
                writer.add_scalar("lr", lr_sched.get_last_lr()[-1], epoch)
                lr_sched.step()

            # saving checkpoints
            if (epoch % self.save_cfg["checkpoints"]) == 0:
                save_cehckpoints_vae(
                    epoch,
                    output_dir,
                    self.data_cfg["dataset_target"],
                    self.cfg["model"],
                    cvae,
                    opt,
                    lr_sched,
                )

            if (epoch % self.hyp_cfg["val_freq"]) == 0:
                cvaes.append(cvae)
                validator.evaluate(
                    cvae, epoch + 1, clustering=clusterer, cond_state=cond_state
                )
                ade, fde = validator.ade_res, validator.fde_res
                metrics["ade"].append(ade)
                metrics["fde"].append(fde)
                if up_topk:
                    metrics["ade_topk"].append(validator.ade_res_topk)
                    metrics["fde_topk"].append(validator.fde_res_topk)
                    metrics["ade_avg"].append(validator.ade_res_avg)
                    metrics["fde_avg"].append(validator.fde_res_avg)
                best_metric_nm = (
                    self.cfg["save"]["best_metric"] + "_topk"
                    if up_topk
                    else self.cfg["save"]["best_metric"]
                )
                ep_bm = (
                    metrics[best_metric_nm].index(min(metrics[best_metric_nm]))
                    if self.cfg["save"]["best_metric"]
                    else -1
                )
                if (
                    self.cfg["save"]["best_metric"]
                    and epoch - ep_bm * cvae_hyp["val_freq"] >= cvae_hyp["patience"]
                ):
                    break
        writer.close()
        test_ep = ep_bm * cvae_hyp["val_freq"] if ep_bm != -1 else -1

        if self.cfg["save"]["objects"]:
            save_objects_vae("cvae", cvaes[ep_bm], test_ep, output_dir)

        results = {k: v[ep_bm] for k, v in metrics.items()}
        # test the generator on test data and save the results
        if testing:
            tester.evaluate(
                cvaes[ep_bm], test_ep, clustering=clusterer, cond_state=cond_state
            )
            results = {"ade": tester.ade_res, "fde": tester.fde_res}
            results.update(
                {
                    "sse": metrics_clusterer["sse"],
                    "clust_score": metrics_clusterer["clust_score"],
                }
            )
            results_line = [f"ade: {tester.ade_res}\n", f"fde: {tester.fde_res} \n"]
            if up_topk:
                results.update(
                    {
                        "ade_topk": tester.ade_res_topk,
                        "fde_topk": tester.fde_res_topk,
                        "ade_avg": tester.ade_res_avg,
                        "fde_avg": tester.fde_res_avg,
                        "wade": tester.wade_res,
                        "wfde": tester.wfde_res,
                        "rank_proposals_accuracy": tester.acc
                    }
                )
                results_line.extend(
                    [
                        f"ade_topk: {tester.ade_res_topk}\n",
                        f"fde_topk: {tester.fde_res_topk} \n",
                        f"ade_avg: {tester.ade_res_avg}\n",
                        f"fde_avg: {tester.fde_res_avg}\n",
                        f"wade: {tester.wade_res} \n",
                        f"wfde: {tester.wfde_res} \n",
                        f"rank_proposals_accuracy: {tester.acc}"
                    ]
                )
        else:
            results_line = [f"{k}: {v[ep_bm]}\n" for k, v in metrics.items()]

        if self.cfg["save"]["final_results"]:
            file = open(os.path.join(output_dir, "final_results.txt"), "w")
            file.writelines(results_line)
            file.close()
        return results, clusterer, cvaes[ep_bm]
