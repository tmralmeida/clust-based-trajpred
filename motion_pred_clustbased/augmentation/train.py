from ..trainers.deterministic import Trainer
from ..trainers.gan_based import GanTrainer
from ..trainers.vae_based import VAETrainer
from ..trainers.vae_clust_based import VAEClusTrainer
from ..trainers.anet import AneTrainer
from ..utils.common import *
from .augment_ds import DataAugmenter
from ..datasets.loaders import *

from torch.utils.data import DataLoader, ConcatDataset
from argparse import ArgumentParser
from yaml import safe_load
from copy import deepcopy


parser = ArgumentParser(description = "Apply sc-gan based augmentation into DL-based forecasters on 2D motion data")

parser.add_argument(
    "--model_cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/training/thor/van_det.yaml",
    required = False,
    help = "configuration file for the network comprising: networks design choices, hyperparameters, etc."
)


parser.add_argument(
    "--synt_gen_cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/clustering/feature_space/thor.yaml",
    required = False,
    help = "Configuration file for the DL-based generator of clusters"
)


parser.add_argument(
    "--aug_cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/augmentation/synth.yaml",
    required = False,
    help = "Augmentation config file"
)


args = parser.parse_args()

model_cfg = load_config(args.model_cfg, "motion_pred_clustbased/cfg/training/default.yaml")
synt_gen_cfg = load_config(args.synt_gen_cfg, "motion_pred_clustbased/cfg/clustering/feature_space/default.yaml")
with open(args.aug_cfg, "rb") as f:
    aug_cfg = safe_load(f)


# assert config files are ok
assert model_cfg['data']['dataset'] == synt_gen_cfg['data']['dataset'], "Cfg files must be wrt the same dataset"
assert synt_gen_cfg['model'] in ["sc-gan", "ft-gan"], "Cfg augmentation file must be based on generators of full trajectories"


# train scgan
gt = GanTrainer(synt_gen_cfg)
results, best_clusterer, gen = gt.train()

# train model -> first results
model_name, use_anet = model_cfg['model'], False
if model_name == "lstm":
    t = Trainer(model_cfg)
elif model_name == "van-gan": # GAN-based models
    t = GanTrainer(model_cfg)
elif model_name  == "van-vae":
    t = VAETrainer(model_cfg)
elif model_name == "vae-clust-based":
    if "anet" not in model_cfg.keys():
        t = VAEClusTrainer(model_cfg)
    else:
        use_anet = True
        t = VAEClusTrainer(model_cfg)
        t_anet = AneTrainer(deepcopy(model_cfg))
else:
    raise NotImplementedError(model_name)

print(f"\n\nTraining first {model_cfg['model']}")

out1 = t.train()
results, model1  = out1[0], out1[-1]
if use_anet:
    out1 = t_anet.train(out1[1], model1, model_cfg['anet'])
    results, model1 = out1[0], out1[1]

train_ds = load_ds("train", model_cfg['data'])
train_dl = DataLoader(train_ds,
                      model_cfg['hyperparameters']['bs'],
                      shuffle = True)

original_samples = get_nsamples(train_dl, gt.cfg, gt.inputs, gt.cs)[0]

da = DataAugmenter(original_samples,
                   aug_cfg,
                   gen,
                   best_clusterer,
                   gt)
final_ds = da.run()
print("Real trajs: {} Generated trajs: {}".format(len(train_ds), len(final_ds)))

if aug_cfg['type'] == "mixed":
    final_ds = ConcatDataset([final_ds, train_ds])

print("Final dataset size", len(final_ds))

train_dl_new = DataLoader(final_ds,
                          model_cfg['hyperparameters']['bs'],
                          shuffle = True)

# TODO: distinction between saved paths

# train model in new train_set -> first results improved?
print(f"\n\nTraining {model_cfg['model']} after augmentation")
out2 = t.train(aug_dataset = train_dl_new)
new_results, model2 = out2[0], out2[-1]
if use_anet:
    out2 = t_anet.train(out2[1], model2, model_cfg['anet'], aug_dataset = train_dl_new)
    new_results, model2 = out2[0], out2[-1]
print("Results by trainining on original dataset", results)
print("Resutls by trainining on synthetic dataset", new_results)