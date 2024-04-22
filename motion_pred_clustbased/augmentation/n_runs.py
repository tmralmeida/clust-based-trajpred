import ray
from torch.utils.data import DataLoader, ConcatDataset
from argparse import ArgumentParser
from yaml import safe_load

from ..trainers.deterministic import Trainer
from ..trainers.gan_based import GanTrainer
from ..trainers.vae_based import VAETrainer
from ..utils.common import *
from .augment_ds import DataAugmenter
from ..datasets.loaders import *
import numpy as np
import datetime

ray.init()


@ray.remote
def train_model(augmenter):
    return augmenter.run()


class AugCapsule:
    def __init__(self, synt_gen_cfg, model_cfg, aug_cfg) -> None:
        self.synt_gen_cfg = synt_gen_cfg
        self.model_cfg = model_cfg
        self.aug_cfg = aug_cfg

    def run(self):
        # train scgan
        gt = GanTrainer(self.synt_gen_cfg)
        results, best_clusterer, gen = gt.train()

        # train model -> first results
        model_name = self.model_cfg["model"]
        if model_name == "lstm":
            t = Trainer(self.model_cfg)
        elif model_name == "van-gan":  # GAN-based models
            t = GanTrainer(self.model_cfg)
        elif model_name == "van-vae":
            t = VAETrainer(self.model_cfg)
        else:
            raise NotImplementedError(model_name)

        print(f"\n\nTraining first {self.model_cfg['model']}")
        results, model1 = t.train()

        train_ds = load_ds("train", self.model_cfg["data"])
        train_dl = DataLoader(
            train_ds, self.model_cfg["hyperparameters"]["bs"], shuffle=True
        )

        original_samples = get_nsamples(train_dl, gt.cfg, gt.inputs, gt.cs)[0]
        da = DataAugmenter(original_samples, self.aug_cfg, gen, best_clusterer, gt)
        final_ds = da.run()
        print("Real trajs: {} Generated trajs: {}".format(len(train_ds), len(final_ds)))

        if self.aug_cfg["type"] == "mixed":
            final_ds = ConcatDataset([final_ds, train_ds])

        print("Final dataset size", len(final_ds))

        train_dl_new = DataLoader(
            final_ds, self.model_cfg["hyperparameters"]["bs"], shuffle=True
        )
        # train model in new train_set -> first results improved?
        print(f"\n\nTraining {self.model_cfg['model']} after augmentation")
        new_results, model2 = t.train(aug_dataset=train_dl_new)

        print("Results by trainining on original dataset", results)
        print("Resutls by trainining on synthetic dataset", new_results)
        return {
            "ade_original_ds": results["ade"],
            "fde_original_ds": results["fde"],
            "ade_synt_ds": new_results["ade"],
            "fde_synt_ds": new_results["fde"],
        }


parser = ArgumentParser(description="n runs of data augmentation")

parser.add_argument(
    "--model_cfg",
    type=str,
    default="motion_pred_clustbased/cfg/training/thor/van_det.yaml",
    required=False,
    help="config file for the network comprising: networks design choices, hyperparameters, etc.",
)


parser.add_argument(
    "--synt_gen_cfg",
    type=str,
    default="motion_pred_clustbased/cfg/clustering/feature_space/thor.yaml",
    required=False,
    help="config file for the DL-based generator of clusters",
)


parser.add_argument(
    "--aug_cfg",
    type=str,
    default="motion_pred_clustbased/cfg/augmentation/synth.yaml",
    required=False,
    help="Augmentation config file",
)


parser.add_argument(
    "--n_runs",
    type=int,
    default=5,
    required=False,
    help="Number of runs",
)


args = parser.parse_args()

model_cfg = load_config(
    args.model_cfg, "motion_pred_clustbased/cfg/training/default.yaml"
)
synt_gen_cfg = load_config(
    args.synt_gen_cfg,
    "motion_pred_clustbased/cfg/clustering/feature_space/default.yaml",
)
with open(args.aug_cfg, "rb") as f:
    aug_cfg = safe_load(f)


# assert config files are ok
assert (
    model_cfg["data"]["dataset"] == synt_gen_cfg["data"]["dataset"]
), "Cfg files must be wrt the same dataset"
assert synt_gen_cfg["model"] in [
    "sc-gan",
    "ft-gan",
], "Cfg augmentation file must be based on generators of full trajectories"

ts = str(datetime.datetime.now().strftime("%y-%m-%d_%a_%H:%M:%S"))

augmenters = [AugCapsule(synt_gen_cfg, model_cfg, aug_cfg) for _ in range(args.n_runs)]

outputs = ray.get([train_model.remote(aug) for aug in augmenters])

ade_prev, fde_prev, ade_synt, fde_synt = [], [], [], []
for run_res in outputs:
    ade_prev.append(run_res["ade_original_ds"])
    fde_prev.append(run_res["fde_original_ds"])
    ade_synt.append(run_res["ade_synt_ds"])
    fde_synt.append(run_res["fde_synt_ds"])

print("==================Overall Results================")
final_results, results_line = [ade_prev, fde_prev, ade_synt, fde_synt], []
for i, k in enumerate(run_res.keys()):
    print(f"{k} {np.mean(final_results[i])}+- {np.std(final_results[i])}")
    results_line.append(f"{k}: {final_results[i]}\n")


file = open(
    os.path.join(
        model_cfg["save"]["path"],
        "outputs",
        f"augmentation_{model_cfg['model']}_n_runs_avg_results_{ts}.txt",
    ),
    "w",
)
file.writelines(results_line)
file.close()
