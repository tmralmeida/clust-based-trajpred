import numpy as np
from copy import deepcopy

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from .utils.common import *
from .trainers.vae_clust_based import VAEClusTrainer
from .trainers.anet import AneTrainer

from argparse import ArgumentParser

parser = ArgumentParser(description = "Grid search on trajectory generative DL models")


parser.add_argument(
    "--cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/training/thor/ours_dist_based.yaml",
    required = False,
    help = "configuration file for the grid search algorithm"
)

parser.add_argument(
    "--n_runs",
    type = int,
    default = 5,
    required = False,
    help = "Number of runs",
)

args = parser.parse_args()
cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/training/default.yaml")


def train(gs_params, cfg):
    update_recursive(cfg, gs_params)
    print("New tested cfg file", cfg['clustering']['kmeans_selfcondgan']['clustering'])
    anet_trainer = "anet" in cfg.keys()
    ades, fdes = [], []
    for _ in range(args.n_runs):
        t = VAEClusTrainer(deepcopy(cfg))
        new_res, clusterer, cvae = t.train()
        print(new_res)
        if anet_trainer:
            t_anet = AneTrainer(deepcopy(cfg))
            print(cfg)
            new_res, _ = t_anet.train(clusterer, cvae, cfg['anet'])
        ade_avg, fde_avg = np.mean(new_res['ade']), np.mean(new_res['fde'])
        for k, v in new_res.items():
            if "topk" in k.lower():
                if "ade" in k.lower():
                    ade_avg = np.mean(v)
                elif "fde" in k.lower():
                    fde_avg = np.mean(v)
        ades.append(ade_avg)
        fdes.append(fde_avg)
    ade_avg, fde_avg =  np.mean(ades), np.mean(fdes)
    print("ADE avg = %1.3f, FDE avg = %1.3f" % (ade_avg, fde_avg))
    tune.report(ade = ade_avg, fde = fde_avg)
    print("Training finished")


def main(cfg, num_samples = 100, max_num_epochs = 100, gpus_per_trial = 1):
    # hyperparam search for scgan
    gs_params = {
        'clustering':
            {
                'kmeans_selfcondgan' :
                    {
                        'clustering' :
                            {
                                'burnin_time' : tune.choice([15,30,45,60,75,100]),
                                'recluster_every' : tune.choice([15,30,45,60,75,100,125,175,200,250,300,400,500]),
                                'weight_cl' : tune.choice([0.0,0.5,0.75,1.0,1.25,1.50,1.75,2.0]),
                            },
                        'generator' :
                            {
                                'condition_type' : tune.choice(["one_hot", "embedding"])

                            },
                        'discriminator' :
                            {
                                'condition_type' : tune.choice(["one_hot", "embedding"])
                            },
                        'hyperparameters':
                            {
                                'g_step_size' : tune.choice([10,20,30,40,50]),
                                'd_step_size' : tune.choice([10,20,30,40,50]),
                                'fm' : tune.choice([True, False]),
                            },
                        'save' :
                            {
                                'best_metric' : tune.choice(['ade', 'silhouette_score', None])
                            }
                    }
            }
    }

    cfg['data']['test'] = False
    params_cols = {}
    for k, v in gs_params.items():
        for sk, vv in v.items():
            if k == "clustering":
                for ssk, vvv in vv.items():
                    for sssk, _ in vvv.items():
                        params_cols[f"{k}/{sk}/{ssk}/{sssk}"] = sssk
            else :
                params_cols[f"{k}/{sk}"] = sk

    scheduler = ASHAScheduler(
        metric = "ade",
        mode = "min",
        max_t = max_num_epochs,
        grace_period = 20
        )

    reporter = CLIReporter(
        parameter_columns = params_cols,
        metric_columns = ["ade", "fde"])


    result = tune.run(
        partial(train, cfg = cfg),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config = gs_params,
        num_samples = num_samples,
        scheduler = scheduler,
        progress_reporter = reporter,
        local_dir = cfg['save']['path'])


    best_trial = result.get_best_trial("ade", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best result: ADE/FDE: {}/{}".format(best_trial.last_result["ade"], best_trial.last_result["fde"]))


if __name__ == "__main__":
    main(cfg, num_samples = 100, max_num_epochs = 100, gpus_per_trial = 0)