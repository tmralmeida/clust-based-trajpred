import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from ..datasets.loaders import *
from .deterministic import Trainer
from ..networks.generators import *
from ..utils.generative import *
from ..utils.deterministic import (
    save_checkpoints_det as save_checkpoints_vae,
    save_objects_det as save_objects_vae,
)
from ..utils.common import *
from ..eval.generative import GenEvaluator


class VAETrainer(Trainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def train(self, *args, **kwargs):
        torch.random.seed()

        runs_dir, output_dir = create_dirs_vae(
            self.cfg, self.cfg["network"], self.hyp_cfg, self.inputs
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

        self.common_settings()
        clusterer, cond_state = None, None
        if self.model_type == "van-vae":
            model = VanillaVAE(self.cfg["network"], self.inp_dim, self.device)
        elif self.model_type == "sup-cvae":
            model = SupCondVAE(self.cfg["network"], self.inp_dim, self.device)
        else:
            raise NotImplementedError(f"{self.cfg['model']} not implemented.")

        # optimizer and scheduler
        opt = optim.Adam(
            model.parameters(), lr=float(self.hyp_cfg["lr"]), weight_decay=1e-4
        )
        lr_sched = optim.lr_scheduler.StepLR(opt, int(self.hyp_cfg["step_size"]))

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

        # metrics
        metrics = {"ade": [], "fde": []}
        up_topk = isinstance(validator.ade, dict)  # topk metrics
        if up_topk:
            metrics.update(
                {
                    "ade_topk": [],
                    "fde_topk": [],
                }
            )

        models = []
        print("===================================" * 5)
        print(f"\nTraining {self.model_type}...\n\n")
        for epoch in range(self.hyp_cfg["max_epochs"]):
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

                    inp_obs, inp_pred, obs, pred = (
                        inp_obs.to(self.device),
                        inp_pred.to(self.device),
                        obs.to(self.device),
                        pred.to(self.device),
                    )

                    loss = train_vae(
                        model,
                        obs,
                        pred,
                        inp_obs,
                        inp_pred,
                        opt,
                        self.cs,
                        self.hyp_cfg,
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
                save_checkpoints_vae(
                    epoch,
                    output_dir,
                    self.data_cfg["dataset_target"],
                    self.cfg["model"],
                    model,
                    opt,
                    lr_sched,
                )

            if (epoch % self.hyp_cfg["val_freq"]) == 0:
                models.append(model)
                validator.evaluate(
                    model, epoch + 1, clustering=clusterer, cond_state=cond_state
                )
                ade, fde = validator.ade_res, validator.fde_res
                metrics["ade"].append(ade)
                metrics["fde"].append(fde)
                if up_topk:
                    metrics["ade_topk"].append(validator.ade_res_topk)
                    metrics["fde_topk"].append(validator.fde_res_topk)
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
                    and epoch - ep_bm * self.hyp_cfg["val_freq"]
                    >= self.hyp_cfg["patience"]
                ):
                    break
        writer.close()
        test_ep = ep_bm * self.hyp_cfg["val_freq"] if ep_bm != -1 else -1

        if self.cfg["save"]["objects"]:
            save_objects_vae(self.model_type, models[ep_bm], test_ep, output_dir)

        results = {k: v[ep_bm] for k, v in metrics.items()}
        # test the generator on test data and save the results
        if testing:
            tester.evaluate(
                models[ep_bm], test_ep, clustering=clusterer, cond_state=cond_state
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
        else:
            results_line = [f"{k}: {v[ep_bm]}\n" for k, v in metrics.items()]

        if self.cfg["save"]["final_results"]:
            file = open(os.path.join(output_dir, "final_results.txt"), "w")
            file.writelines(results_line)
            file.close()
        return results, clusterer, models[ep_bm]
