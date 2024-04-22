from copy import deepcopy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import davies_bouldin_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from ..datasets.loaders import *
from .deterministic import Trainer
from ..networks.generators import *
from ..networks.discriminators import *
from ..utils.generative import *
from ..utils.common import *
from ..eval.generative import GenEvaluator


class GanTrainer(Trainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def train(self, *args, **kwargs):
        torch.random.seed()
        float_dtype = torch.FloatTensor

        if "aug_dataset" in kwargs:
            self.train_dl_d = kwargs["aug_dataset"]
            self.train_dl_g = kwargs["aug_dataset"]
            print(
                "\n\n=======================================================Training new model on synthetic dataset=======================================================\n\n"
            )
        else:
            train_ds = load_ds("train", self.data_cfg)
            self.train_dl_d = DataLoader(train_ds, self.hyp_cfg["bs"], shuffle=True)
            self.train_dl_g = DataLoader(train_ds, self.hyp_cfg["bs"], shuffle=True)

        self.common_settings()

        gen_cfg = self.cfg["generator"]
        disc_cfg = self.cfg["discriminator"]

        clusterer, cond_state = None, None
        #  models definition
        if self.model_type == "van-gan":
            # generator
            gen = VanillaGenerator(gen_cfg, self.inp_dim, self.device)
            # discriminator
            disc = (
                FFVanDiscriminator(disc_cfg, self.inp_dim, self.device)
                if disc_cfg["net_type"] == "ff"
                else TemplVanDiscriminator(disc_cfg, self.inp_dim, self.device)
            )
            runs_dir, self.output_dir = create_dirs(
                self.cfg, self.inputs
            )  # create output files
        elif self.model_type == "sup-cgan":
            gen = SupCondGenerator(gen_cfg, self.inp_dim, self.device)
            disc = (
                FFSupCondDiscriminator(disc_cfg, self.inp_dim, self.device)
                if disc_cfg["net_type"] == "ff"
                else TempSupCondDiscriminator(disc_cfg, self.inp_dim, self.device)
            )
            assert (
                gen.k_value == disc.k_value
            ), "Please chang cfg file. Disc and Gen dhould have the same number of supervised labels"
            print(f"This data set has {gen.k_value} supervised labels!")
            runs_dir, self.output_dir = create_dirs(
                self.cfg, self.inputs
            )  # create output files
        elif self.model_type == "sc-gan":
            condition_cfg = gen_cfg["condition"]
            self.cfg["clustering"]["kwargs"]["n_clusters"] = (
                args[0]
                if len(args) > 0 and args[0] is not None
                else self.cfg["clustering"]["kwargs"]["n_clusters"]
            )
            clustering_info = self.cfg["clustering"]
            disc_cfg["n_labels"], gen_cfg["n_labels"] = (
                clustering_info["kwargs"]["n_clusters"],
                clustering_info["kwargs"]["n_clusters"],
            )
            clustering_name = clustering_info["name"]

            # discriminator
            disc = (
                FFSoftCondDiscriminator(
                    disc_cfg, self.inp_dim if self.inp_dim > 0 else 2, self.device
                )
                if disc_cfg["net_type"] == "ff"
                else TempSoftCondDiscriminator(
                    disc_cfg, self.inp_dim if self.inp_dim > 0 else 2, self.device
                )
            )

            # generator
            gen = (
                SoftCondGenerator(gen_cfg, self.inp_dim, self.device)
                if self.inp_dim > 0
                else FTSoftCondGenerator(gen_cfg, self.device)
            )
            print(
                f"This model is conditioned through {gen_cfg['condition']} and {disc.k_value} unsupervised labels."
            )

            x_cluster, lbls = get_nsamples(
                self.train_dl_d, self.cfg, self.inputs, self.cs
            )
            clusterer = get_clustering(clustering_name)(
                discriminator=disc,
                x_cluster=x_cluster,
                batch_size=self.hyp_cfg["bs"],
                **clustering_info["kwargs"],
            )

            print(
                "Initializing new clustering. The first clustering can be quite slow."
            )
            clusterer.recluster(discriminator=disc)
            runs_dir, self.output_dir = create_dirs(
                self.cfg, self.inputs
            )  # create output files
            np.savez(
                os.path.join(self.output_dir, "clusters", "cluster_samples.npz"),
                x=x_cluster,
            )
            if self.ds == "argoverse" or self.ds == "thor":
                lbls = lbls.unsqueeze(dim=1).repeat(1, x_cluster.size(1), 1)
                samples_labels = torch.cat([x_cluster, lbls], dim=-1)
                np.savez(
                    os.path.join(self.output_dir, "clusters", "labels.npz"),
                    data=samples_labels,
                )
            cond_state = ConditionState(self.model_type, self.cs, clusterer, condition_cfg)

        elif self.model_type == "ft-gan":  # full trajectory gan
            # generator
            gen = FTGenerator(gen_cfg, self.inp_dim, self.device)

            # discriminator
            disc = (
                FFVanDiscriminator(disc_cfg, 2, self.device)
                if disc_cfg["net_type"] == "ff"
                else TemplVanDiscriminator(disc_cfg, self.inp_dim, self.device)
            )
            runs_dir, self.output_dir = create_dirs(
                self.cfg, self.inputs
            )  # create output files
        else:
            raise NotImplementedError(f"{self.cfg['model']} not implemented.")

        writer = SummaryWriter(runs_dir)
        gen.type(float_dtype).train()
        disc.type(float_dtype).train()

        # optimizers and schedulers
        g_opt = optim.Adam(
            gen.parameters(), lr=float(self.hyp_cfg["g_lr"]), weight_decay=1e-4
        )
        d_opt = optim.Adam(
            disc.parameters(), lr=float(self.hyp_cfg["d_lr"]), weight_decay=1e-4
        )
        g_lr_sched = optim.lr_scheduler.StepLR(g_opt, int(self.hyp_cfg["g_step_size"]))
        d_lr_sched = optim.lr_scheduler.StepLR(d_opt, int(self.hyp_cfg["d_step_size"]))

        # validation and testing objects
        validator = GenEvaluator(
            self.val_dl, self.cfg, self.inputs, self.outputs, self.device, writer=writer
        )

        testing = False
        if self.data_cfg["test"]:
            test_ds = load_ds("test", self.data_cfg)
            testing = True
            test_dl = DataLoader(test_ds, self.hyp_cfg["bs"])
            tester = GenEvaluator(
                test_dl, self.cfg, self.inputs, self.outputs, self.device
            )

        # best metric
        if self.cfg["save"]["best_metric"] in ["clust_score", "sse"]:
            assert (
                self.model_type == "sc-gan"
            ), f"Not possible to use {self.cfg['save']['best_metric']} for {self.model_type}"

        # metrics for top 1
        metrics = {"ade": [], "fde": []}
        up_topk = isinstance(validator.ade, dict)  # topk metrics
        if up_topk:
            metrics.update(
                {
                    "ade_topk": [],
                    "fde_topk": [],
                }
            )

        if self.model_type == "sc-gan":
            if clustering_name == "kmeans_selfcondgan":
                cl_metname = "sse"  # inertia
                features = clusterer.get_features(x_cluster)
                labels = clusterer.get_labels(x_cluster, None)
                writer.add_embedding(
                    features, metadata=labels, global_step=0, tag="feature space"
                )
                metrics.update({"sse": [], "clust_score": []})

        # handling nsteps optimization
        assert (
            self.hyp_cfg["d_nsteps"] >= self.hyp_cfg["g_nsteps"]
            and self.hyp_cfg["d_nsteps"] > 0
            and self.hyp_cfg["g_nsteps"] > 0
        ), "number of disciminator's steps MUST BE at least equals to the generator's steps!"
        n_steps_epoch = int(
            len(self.train_dl_g) * (self.hyp_cfg["d_nsteps"] / self.hyp_cfg["g_nsteps"])
        ) + len(
            self.train_dl_g
        )  # number of steps of the discriminator

        gens, discs, clusterers, gstep, proc_labels = [], [], [], 0, None
        # training loop
        for epoch in range(self.hyp_cfg["max_epochs"]):
            g_losses, d_losses = [], []
            train_dl_iterd = iter(self.train_dl_d)
            train_dl_iterg = iter(self.train_dl_g)
            curr_steps_g, curr_steps_d = 0, 0

            with tqdm(total=len(self.train_dl_g)) as tepoch:
                for _ in range(n_steps_epoch):
                    tepoch.set_description(f"Epoch gen {epoch + 1}")
                    if (
                        curr_steps_g < self.hyp_cfg["g_nsteps"]
                    ):  # iterating over gen' data loader
                        curr_steps_g += 1
                        tepoch.update()
                        try:
                            batch_g = next(train_dl_iterg)
                        except StopIteration:
                            train_dl_iterg = iter(self.train_dl_g)
                            batch_g = next(train_dl_iterg)
                        (
                            obs_g,
                            pred_g,
                            inp_obs_g,
                            inp_pred_g,
                            ft_g,
                            proc_labels_g,
                        ) = get_batch(
                            batch_g,
                            self.cs,
                            self.model_type,
                            self.ds,
                            self.inputs,
                            self.device,
                            clustering=clusterer,
                            gen=gen,
                            cond_state=cond_state,
                            output_type=self.data_cfg["output"],
                        )
                        g_loss = train_generator(
                            disc,
                            gen,
                            obs_g,
                            pred_g,
                            inp_obs_g,
                            inp_pred_g,
                            g_opt,
                            self.cs,
                            self.hyp_cfg,
                            proc_labels_g,
                            self.device,
                        )
                        g_losses.append(g_loss)
                        gavg_loss = sum(g_losses) / len(g_losses)
                        # reclustering
                        if (self.model_type == "sc-gan") and (
                            gstep % clustering_info["recluster_every"] == 0
                            and gstep > clustering_info["burnin_time"]
                        ):
                            clusterer.x = x_cluster
                            clusterer.recluster(
                                discriminator=disc, x_batch=ft_d
                            )  # x_batch is for online methods
                            features = clusterer.get_features(x_cluster)
                            labels = clusterer.get_labels(x_cluster, None)
                            writer.add_embedding(
                                features,
                                metadata=labels,
                                global_step=gstep,
                                tag="feature space",
                            )
                        gstep += 1
                    if (
                        curr_steps_d < self.hyp_cfg["d_nsteps"]
                    ):  # iterating over disc' data loader
                        curr_steps_d += 1
                        try:
                            batch_d = next(train_dl_iterd)
                            (
                                obs_d,
                                pred_d,
                                inp_obs_d,
                                inp_pred_d,
                                ft_d,
                                proc_labels_d,
                            ) = get_batch(
                                batch_d,
                                self.cs,
                                self.model_type,
                                self.ds,
                                self.inputs,
                                self.device,
                                clustering=clusterer,
                                gen=gen,
                                cond_state=cond_state,
                                output_type=self.data_cfg["output"],
                            )
                        except StopIteration:
                            train_dl_iterd = iter(self.train_dl_d)
                            batch_d = next(train_dl_iterd)
                            (
                                obs_d,
                                pred_d,
                                inp_obs_d,
                                inp_pred_d,
                                ft_d,
                                proc_labels_d,
                            ) = get_batch(
                                batch_d,
                                self.cs,
                                self.model_type,
                                self.ds,
                                self.inputs,
                                self.device,
                                clustering=clusterer,
                                gen=gen,
                                cond_state=cond_state,
                                output_type=self.data_cfg["output"],
                            )
                        d_loss = train_discriminator(
                            disc,
                            gen,
                            obs_d,
                            inp_obs_d,
                            inp_pred_d,
                            d_opt,
                            self.hyp_cfg["clip_thresh_d"],
                            proc_labels_d,
                            self.cs,
                            self.device,
                        )
                        d_losses.append(d_loss)
                        davg_loss = sum(d_losses) / len(d_losses)
                    else:
                        curr_steps_g, curr_steps_d = 0, 0

                    if self.model_type == "sc-gan":
                        tepoch.set_postfix(
                            {
                                "gl": gavg_loss,
                                "dl": davg_loss,
                                cl_metname: clusterer.method.inertia_,
                            }
                        )
                    else:  # vanilla or cgan
                        tepoch.set_postfix({"gl": gavg_loss, "dl": davg_loss})
            writer.add_scalar("gloss_avg", gavg_loss, epoch)
            writer.add_scalar("dloss_avg", davg_loss, epoch)
            writer.add_scalar("gen_lr", g_lr_sched.get_last_lr()[-1], epoch)
            writer.add_scalar("disc_lr", d_lr_sched.get_last_lr()[-1], epoch)
            g_lr_sched.step()
            d_lr_sched.step()
            if self.model_type == "sup-cgan":
                disc.eval()
                with torch.no_grad():
                    features_sup, labels_sup = [], []
                    for batch_ in self.train_dl_d:
                        obs, pred, inp_obs, inp_pred, _, proc_labels = get_batch(
                            batch_,
                            self.cs,
                            self.model_type,
                            self.ds,
                            self.inputs,
                            self.device,
                            clustering=clusterer,
                            gen=gen,
                        )
                        feat = disc(
                            concat_subtrajs(inp_obs, inp_pred),
                            get_features=True,
                            labels=proc_labels,
                        )
                        features_sup.append(feat)
                        labels_sup.append(proc_labels)
                labels_sup = torch.cat(labels_sup, dim=0)
                features_sup = torch.cat(features_sup, dim=0)
                writer.add_embedding(
                    features_sup,
                    metadata=labels_sup,
                    global_step=epoch,
                    tag="feature space",
                )
                # saving checkpoints
            if (
                self.save_cfg["checkpoints"] != -1
                and (epoch % self.save_cfg["checkpoints"]) == 0
            ):
                save_checkpoints(
                    epoch,
                    self.output_dir,
                    self.data_cfg["dataset_target"],
                    self.cfg["model"],
                    gen,
                    disc,
                    g_opt,
                    d_opt,
                    g_lr_sched,
                    d_lr_sched,
                    clustering=clusterer,
                )

            # validation
            if (epoch % self.hyp_cfg["val_freq"]) == 0:
                gens.append(gen)
                discs.append(disc)
                validator.evaluate(
                    gen, epoch + 1, clustering=clusterer, cond_state=cond_state
                )
                ade, fde = validator.ade_res, validator.fde_res
                metrics["ade"].append(ade)
                metrics["fde"].append(fde)
                if up_topk:
                    metrics["ade_topk"].append(validator.ade_res_topk)
                    metrics["fde_topk"].append(validator.fde_res_topk)
                if self.model_type == "sc-gan":
                    clusterers.append(clusterer)
                    dbi_avg = davies_bouldin_score(
                        clusterer.get_features(clusterer.x).numpy(),
                        clusterer.get_labels(clusterer.x, None).numpy(),
                    )
                    metrics["sse"].append(clusterer.method.inertia_)
                    metrics["clust_score"].append(dbi_avg)
                best_metric_nm = (
                    str(self.cfg["save"]["best_metric"]) + "_topk"
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
                    and epoch - ep_bm * self.hyp_cfg["val_freq"]
                    >= self.hyp_cfg["patience"]
                ):
                    break
        writer.close()
        test_ep = ep_bm * self.hyp_cfg["val_freq"] if ep_bm != -1 else -1
        best_clusterer = clusterers[ep_bm] if self.model_type == "sc-gan" else None
        best_generator = gens[ep_bm]
        # save results, generator, discriminator, clusterer, and anet
        if self.cfg["save"]["objects"]:
            save_objects(
                best_generator,
                discs[ep_bm],
                output_dir=self.output_dir,
                clustering=best_clusterer,
            )

        results = {k: v[ep_bm] for k, v in metrics.items()}
        # test the generator on test data and save the results
        if testing:
            tester.evaluate(
                best_generator,
                test_ep,
                clustering=best_clusterer,
                cond_state=cond_state,
            )
            results = {"ade": tester.ade_res, "fde": tester.fde_res}
            results_line = [f"ade: {tester.ade_res}\n", f"fde: {tester.fde_res} \n"]
            if up_topk:
                results.update(
                    {
                        "ade_topk": tester.ade_res_topk,
                        "fde_topk": tester.fde_res_topk,
                        "wade": tester.wade_res,
                        "wfde": tester.wfde_res,
                    }
                )
                results_line.extend(
                    [
                        f"ade_topk: {tester.ade_res_topk}\n",
                        f"fde_topk: {tester.fde_res_topk} \n",
                        f"wade: {tester.wade_res} \n",
                        f"wfde: {tester.wfde_res} \n",
                    ]
                )
            if self.model_type == "sc-gan":
                results_line += [
                    f"sse: {metrics['sse'][ep_bm]} \n",
                    f"clust_score: {metrics['clust_score'][ep_bm]} \n",
                ]
                results.update(
                    {
                        "sse": metrics["sse"][ep_bm],
                        "clust_score": metrics["clust_score"][ep_bm],
                    }
                )
        else:
            results_line = [f"{k}: {v[ep_bm]}\n" for k, v in metrics.items()]

        if self.cfg["save"]["final_results"]:
            file = open(os.path.join(self.output_dir, "final_results.txt"), "w")
            file.writelines(results_line)
            file.close()
        return results, best_clusterer, best_generator


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train GAN-based forecasters on 2D motion data")

    parser.add_argument(
        "--cfg",
        type=str,
        default="motion_pred_clustbased/cfg/clustering/feature_space/thor.yaml",
        required=False,
        help="configuration file comprising: networks design choices, hyperparameters, etc.",
    )

    args = parser.parse_args()
    cfg = load_config(
        args.cfg, "motion_pred_clustbased/cfg/clustering/feature_space/default.yaml"
    )
    cfg["hyperparameters"]["max_epochs"] = 50
    gt = GanTrainer(cfg)
    gt.train()
    print("passed")
