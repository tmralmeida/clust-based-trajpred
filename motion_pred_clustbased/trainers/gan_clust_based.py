import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import davies_bouldin_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from ..datasets.loaders import *
from .vae_clust_based import VAEClusTrainer
from ..networks.generators import SupCondGenerator
from ..networks.discriminators import FFSupCondDiscriminator, TempSupCondDiscriminator
from ..utils.generative import *
from ..utils.common import *
from ..eval.generative import GenEvaluator


class GANClusTrainer(VAEClusTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def train(self, *args, **kwargs):
        torch.random.seed()
        float_dtype = torch.FloatTensor

        train_ds = load_ds("train", self.data_cfg)

        # 1st step get clusterer object
        assert self.model_type == "gan-clust-based", "Wrong model"
        print("\nGetting clustering...")
        gan_hyp = self.cfg["cgan"]["hyperparameters"]
        self.train_dl_d = DataLoader(train_ds, gan_hyp["bs"], shuffle=True)
        self.train_dl_g = DataLoader(train_ds, gan_hyp["bs"], shuffle=True)
        dataset_np = self.get_ds_np()
        self.cfg["cgan"]["model"] = self.model_type
        self.cfg["cgan"]["data"] = self.data_cfg
        self.cfg["cgan"]["save"] = self.save_cfg
        runs_dir, self.output_dir = create_dirs(
            self.cfg["cgan"], self.inputs
        )  # create output files
        writer = SummaryWriter(runs_dir)

        clust_algo = self.clusterer_info["algorithm"]
        ncl = (
            self.clusterer_info[clust_algo]["n_clusters"]
            if clust_algo in ["k-means", "k-shape"]
            else self.clusterer_info[clust_algo]["clustering"]["kwargs"]["n_clusters"]
        )
        ncl = ncl if len(args) == 0 else args[0]
        metrics_clusterer, clusterer = self.get_clusterer(
            dataset_np, ncl, out_path=self.output_dir, writer=writer
        )
        self.cfg["cgan"]["generator"]["n_labels"] = clusterer.k
        self.cfg["cgan"]["discriminator"]["n_labels"] = clusterer.k
        self.common_settings()

        gen_cfg = self.cfg["cgan"]["generator"]
        disc_cfg = self.cfg["cgan"]["discriminator"]
        gen = SupCondGenerator(gen_cfg, self.inp_dim, self.device)
        disc = (
            FFSupCondDiscriminator(disc_cfg, self.inp_dim, self.device)
            if disc_cfg["net_type"] == "ff"
            else TempSupCondDiscriminator(disc_cfg, self.inp_dim, self.device)
        )

        # conditioning handler
        cond_state = ConditionState(
            self.model_type,
            self.cs,
            clusterer,
            self.cfg["cgan"]["condition"],
        )

        writer = SummaryWriter(runs_dir)
        gen.type(float_dtype).train()
        disc.type(float_dtype).train()

        # optimizers and schedulers
        g_opt = optim.Adam(
            gen.parameters(), lr=float(gan_hyp["g_lr"]), weight_decay=1e-4
        )
        d_opt = optim.Adam(
            disc.parameters(), lr=float(gan_hyp["d_lr"]), weight_decay=1e-4
        )
        g_lr_sched = optim.lr_scheduler.StepLR(g_opt, int(gan_hyp["g_step_size"]))
        d_lr_sched = optim.lr_scheduler.StepLR(d_opt, int(gan_hyp["d_step_size"]))

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
            test_dl = DataLoader(test_ds, gan_hyp["bs"])
            tester = GenEvaluator(
                test_dl,
                self.cfg,
                self.inputs,
                self.outputs,
                self.device,
                n_clusters=ncl,
            )

        # best metric
        metrics = {"ade": [], "fde": []}
        metrics.update(
            {
                "sse": [metrics_clusterer["sse"]] * gan_hyp["max_epochs"],
                "clust_score": [metrics_clusterer["clust_score"]]
                * gan_hyp["max_epochs"],
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

        # handling nsteps optimization
        assert (
            gan_hyp["d_nsteps"] >= gan_hyp["g_nsteps"]
            and gan_hyp["d_nsteps"] > 0
            and gan_hyp["g_nsteps"] > 0
        ), "number of disciminator's steps MUST BE at least equals to the generator's steps!"
        n_steps_epoch = int(
            len(self.train_dl_g) * (gan_hyp["d_nsteps"] / gan_hyp["g_nsteps"])
        ) + len(
            self.train_dl_g
        )  # number of steps of the discriminator

        gens, discs = [], []
        print("===================================" * 5)
        print("\nTraining cGAN...\n\n")
        # training loop
        for epoch in range(gan_hyp["max_epochs"]):
            g_losses, d_losses = [], []
            train_dl_iterd = iter(self.train_dl_d)
            train_dl_iterg = iter(self.train_dl_g)
            curr_steps_g, curr_steps_d = 0, 0

            with tqdm(total=len(self.train_dl_g)) as tepoch:
                for _ in range(n_steps_epoch):
                    tepoch.set_description(f"Epoch gen {epoch + 1}")
                    if (
                        curr_steps_g < gan_hyp["g_nsteps"]
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
                            cond_state=self.cs,
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
                            gan_hyp,
                            proc_labels_g,
                            self.device,
                        )
                        g_losses.append(g_loss)
                        gavg_loss = sum(g_losses) / len(g_losses)
                    if (
                        curr_steps_d < gan_hyp["d_nsteps"]
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
                            gan_hyp["clip_thresh_d"],
                            proc_labels_d,
                            self.cs,
                            self.device,
                        )
                        d_losses.append(d_loss)
                        davg_loss = sum(d_losses) / len(d_losses)
                    else:
                        curr_steps_g, curr_steps_d = 0, 0
                    tepoch.set_postfix({"gl": gavg_loss, "dl": davg_loss})
            writer.add_scalar("gloss_avg", gavg_loss, epoch)
            writer.add_scalar("dloss_avg", davg_loss, epoch)
            writer.add_scalar("gen_lr", g_lr_sched.get_last_lr()[-1], epoch)
            writer.add_scalar("disc_lr", d_lr_sched.get_last_lr()[-1], epoch)
            g_lr_sched.step()
            d_lr_sched.step()
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
            if (epoch % gan_hyp["val_freq"]) == 0:
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
                    metrics["ade_avg"].append(validator.ade_res_avg)
                    metrics["fde_avg"].append(validator.fde_res_avg)
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
                    and epoch - ep_bm * gan_hyp["val_freq"] >= gan_hyp["patience"]
                ):
                    break
        writer.close()
        test_ep = ep_bm * gan_hyp["val_freq"] if ep_bm != -1 else -1
        best_generator = gens[ep_bm]
        if self.cfg["save"]["objects"]:
            save_objects(
                best_generator,
                discs[ep_bm],
                output_dir=self.output_dir,
                clustering=None,
            )
        results = {k: v[ep_bm] for k, v in metrics.items()}
        # test the generator on test data and save the results
        if testing:
            tester.evaluate(
                best_generator, test_ep, clustering=clusterer, cond_state=cond_state
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
                        "rank_proposals_accuracy": tester.acc,
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
                        f"rank_proposals_accuracy: {tester.acc}",
                    ]
                )
        else:
            results_line = [f"{k}: {v[ep_bm]}\n" for k, v in metrics.items()]

        if self.cfg["save"]["final_results"]:
            file = open(os.path.join(self.output_dir, "final_results.txt"), "w")
            file.writelines(results_line)
            file.close()
        return results, clusterer, best_generator
