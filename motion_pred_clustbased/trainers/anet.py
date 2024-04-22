import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from ..networks.deterministic import ANet
from ..datasets.loaders import *
from .deterministic import Trainer
from ..networks.generators import *
from ..utils.common import *
from ..utils.deterministic import *
from ..eval.generative import *


class AneTrainer(Trainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def train(self, clusterer, gen_model, anet_cfg, *args, **kwargs):
        gen_model.eval()
        runs_dir, output_dir = create_dirs_anet(
            self.cfg, self.inputs
        )  # create output files
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
        self.cfg["model"] = "anet"

        n_clusters = clusterer.k
        anet_cfg["network"]["inp_features"] = self.inp_dim
        anet_hyp = anet_cfg["hyperparameters"]

        # Anet model
        anet = ANet(anet_cfg["network"], n_clusters, self.device)

        # loss function
        criterion = nn.CrossEntropyLoss()

        # optimizer
        opt = optim.Adam(anet.parameters(), lr=float(anet_hyp["lr"]))

        # scheduler
        lr_sched = optim.lr_scheduler.StepLR(
            opt, int(anet_hyp["step_size"])
        )  # TODO reduce on Plateau -> accuracy?

        # validation and testing objects
        validator = GenEvaluator(
            self.val_dl,
            self.cfg,
            self.inputs,
            self.outputs,
            self.device,
            writer=writer,
            n_clusters=n_clusters,
        )

        testing = False
        if self.data_cfg["test"]:
            test_ds = load_ds("test", self.data_cfg)
            testing = True
            test_dl = DataLoader(test_ds, anet_hyp["bs"])
            tester = GenEvaluator(
                test_dl,
                self.cfg,
                self.inputs,
                self.outputs,
                self.device,
                n_clusters=n_clusters,
            )

        # metrics
        up_topk = isinstance(validator.ade, dict)  # topk metrics
        metrics = {"ade": [], "fde": [], "acc": []}
        if up_topk:
            metrics.update(
                {
                    "ade_topk": [],
                    "fde_topk": [],
                    "ade_avg": [],
                    "fde_avg": [],
                }
            )

        anets = []
        print("Training ANet")
        for epoch in range(anet_hyp["max_epochs"]):
            anet_losses, anet_accs = [], []
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

                    if anet_cfg["network"]["inp_seq_len"] == (
                        self.data_cfg["obs_len"] - 1
                    ):  # anet's input: observed trajectory
                        inp_net = inp_obs
                    elif (
                        anet_cfg["network"]["inp_seq_len"] == self.data_cfg["pred_len"]
                    ):  # anet's input: future trajectory
                        if anet_cfg["network"]["mode"] == "multiclass_gen":
                            predicted_traj = gen_model(inp_obs, labels=proc_label)
                            if isinstance(predicted_traj, tuple):
                                predicted_traj = predicted_traj[0]
                            inp_net = predicted_traj.clone()
                        elif anet_cfg["network"]["mode"] == "multiclass":
                            inp_net = inp_pred
                        elif anet_cfg["network"]["mode"] == "multiclass_concat":
                            inp_net = []
                            for i in range(clusterer.k):
                                pp_cl = torch.full((inp_obs.size(0),), i).long()
                                fake_preds = gen_model(inp_obs, labels=pp_cl)
                                if isinstance(fake_preds, tuple):
                                    fake_preds = fake_preds[0]
                                inp_net.append(fake_preds)
                            inp_net = torch.stack(inp_net, dim=1)
                    elif anet_cfg["network"]["inp_seq_len"] == (
                        self.data_cfg["obs_len"] + self.data_cfg["pred_len"] - 1
                    ):  # anet's input: full trajectory
                        if anet_cfg["network"]["mode"] == "multiclass_gen":  # generated
                            predicted_traj = gen_model(inp_obs, labels=proc_label)
                            if isinstance(predicted_traj, tuple):
                                predicted_traj = predicted_traj[0]
                            inp_net = torch.cat(
                                [inp_obs, predicted_traj.clone()], dim=1
                            )
                        elif anet_cfg["network"]["mode"] == "multiclass":
                            inp_net = torch.cat([inp_obs, inp_pred], dim=1)
                        elif anet_cfg["network"]["mode"] == "multiclass_concat":
                            inp_net = []
                            for i in range(clusterer.k):
                                pp_cl = torch.full((inp_obs.size(0),), i).long()
                                fake_preds = gen_model(inp_obs, labels=pp_cl)
                                if isinstance(fake_preds, tuple):
                                    fake_preds = fake_preds[0]
                                inp_net.append(concat_subtrajs(inp_obs, fake_preds))
                            inp_net = torch.stack(inp_net, dim=1)

                    anet.train()
                    out = anet(inp_net)
                    loss = criterion(out, proc_label).mean()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    anet_losses.append(loss)
                    anetavg_loss = sum(anet_losses) / len(anet_losses)
                    anet_acc = (
                        out.argmax(dim=-1) == proc_label
                    ).int().sum() / out.size(0)
                    anet_accs.append(anet_acc)
                    anet_avg_acc = sum(anet_accs) / len(anet_accs)
                    tepoch.set_postfix(
                        {"al": anetavg_loss.item(), "train_acc": anet_avg_acc.item()}
                    )

                lr_sched.step()
                if (epoch % anet_hyp["val_freq"]) == 0:
                    anets.append(anet)
                    validator.evaluate(gen_model, epoch + 1, clustering=clusterer, anet=anet)
                    metrics["ade"].append(validator.ade_res)
                    metrics["fde"].append(validator.fde_res)
                    metrics["acc"].append(validator.acc)
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
                    and epoch - ep_bm * anet_hyp["val_freq"] >= anet_hyp["patience"]
                ):
                    break
        writer.close()
        test_ep = ep_bm * anet_hyp["val_freq"] if ep_bm != -1 else -1
        if self.cfg["save"]["objects"]:
            torch.save(anets[ep_bm].state_dict(), os.path.join(output_dir, "anet.pth"))

        # test the generator on test data and save the results
        results = {k: v[ep_bm] for k, v in metrics.items()}
        if testing:
            tester.evaluate(gen_model, test_ep, clustering=clusterer, anet=anet)
            results = {"ade": tester.ade_res, "fde": tester.fde_res}
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
            file = open(os.path.join(output_dir, "final_results.txt"), "w")
            file.writelines(results_line)
            file.close()

        return results, anets[ep_bm]
