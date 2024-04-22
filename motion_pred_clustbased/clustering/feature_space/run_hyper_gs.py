import numpy as np
from copy import deepcopy

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from ...utils.common import *
from ...trainers.gan_based import GanTrainer
from argparse import ArgumentParser


parser = ArgumentParser(description = "Grid search on clustering based on SC-GAN")


parser.add_argument(
    "--cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/clustering/feature_space/thor.yaml",
    required = False,
    help = "configuration file comprising: networks design choices, hyperparameters, etc."
)

parser.add_argument(
    "--n_runs",
    type = int,
    default = 5,
    required = False,
    help = "Number of runs",
)

args = parser.parse_args()
cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/clustering/feature_space/default.yaml")


def train(gs_params, cfg):
    update_recursive(cfg, gs_params)
    assert cfg["model"] == "sc-gan", "Elbow method only possible for clustering-related methods"
    assert cfg["generator"]["condition"]["train"] == "recognizer" and cfg["generator"]["condition"]["inference"] == "recognizer", "Elbow method recquires one hot labels during training and inference"
    print("New tested cfg file", cfg['clustering']['kmeans_selfcondgan']['clustering'])
    dbis, ades, fdes = [], [], []
    for _ in range(args.n_runs):
        t = GanTrainer(cfg)
        results, _, _ = t.train()
        dbis.append(results['clust_score'])
        ades.append(results['ade'])
        fdes.append(results['fde'])
    avg_dbi, std_dbi, avg_ade, std_ade, avg_fde, std_fde = np.mean(dbis), np.std(dbis), np.mean(ades), np.std(ades), np.mean(fdes), np.std(fdes)
    print("DBI avg = %1.3f, DBI std = %1.3f" % (avg_dbi, std_dbi))
    print("ADE avg = %1.3f, ADE std = %1.3f" % (avg_ade, std_ade))
    print("FDE avg = %1.3f, FDE std = %1.3f" % (avg_fde, std_fde))
    tune.report(dbi = avg_dbi, ade = avg_ade, fde = avg_fde)
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
                                # 'weight_cl' : tune.choice([0.0,0.5,0.75,1.0,1.25,1.50,1.75,2.0]),
                                'kwargs' :
                                    {
                                        'n_clusters' : tune.choice([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
                                    }
                            },
                        'discriminator' :
                                    {
                                        'layer_ret_feat' : tune.choice(["first", "second", "third", "fourth", "fifth"])
                                    },
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

    scheduler = ASHAScheduler(metric = "dbi",
                              mode = "min",
                              max_t = max_num_epochs,
                              grace_period = 20)

    reporter = CLIReporter(
        parameter_columns = params_cols,
        metric_columns = ["dbi", "ade", "fde"])

    result = tune.run(
        partial(train, cfg = cfg),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config = gs_params,
        num_samples = num_samples,
        scheduler = scheduler,
        progress_reporter = reporter,
        local_dir = cfg['save']['path'])


    best_trial = result.get_best_trial("dbi", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best result: DBI: {}".format(best_trial.last_result["dbi"]))
    print("Best result: ADE/FDE: {}/{}".format(best_trial.last_result["ade"], best_trial.last_result["fde"]))


if __name__ == "__main__":
    main(cfg, num_samples = 200, max_num_epochs = 100, gpus_per_trial = 0)

