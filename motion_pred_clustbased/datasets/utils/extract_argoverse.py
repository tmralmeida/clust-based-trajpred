import yaml
import os
import numpy as np
import pandas as pd
import _pickle as cPickle
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser(description = "Extracting Argoverse data set")
parser.add_argument(
    "--cfg",
    type = str,
    default = "motion_pred_clustbased/cfg/preprocessing/argoverse.yaml",
    required = False,
    help = "configuration file comprising extraction/preprocessing information"
)

args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)


extraction_info = cfg["extraction"]
root_dir = extraction_info["root_dir"]
root_train = os.path.join(root_dir, "forecasting_train_v1.1/train/data/") # path to train files
root_val = os.path.join(root_dir, "forecasting_val_v1.1/val/data/") # path to val files
roots = [root_train, root_val]
save_dir = extraction_info["save_dir"]
full_traj_size = cfg["obs_len"] + cfg["pred_len"]


if not os.path.exists(save_dir):
    print(f"Creating {save_dir}")
    os.makedirs(save_dir)


#  sets info
tr_set_info = extraction_info["sets"]["train"] # info about train set
val_set_info = extraction_info["sets"]["val"] # info about train set
test_set_info = extraction_info["sets"]["test"] # info about train set


save_dfs = ["train.csv", "val.csv", "test.csv"]
save_lists = ["train.pkl", "val.pkl", "test.pkl"]


av_targets_size = [tr_set_info["av"], val_set_info["av"], test_set_info["av"]]
agent_targets_size = [tr_set_info["agent"], val_set_info["agent"], test_set_info["agent"]]
others_targets_size = [tr_set_info["others"], val_set_info["others"], test_set_info["others"]]

# logging
print("\n ===== Data set composition ===== \n")
for s in range(len(av_targets_size)):
    print("%s composed of: %d av, %d agent, %d others" %
          (save_lists[s].split('.')[0],
           av_targets_size[s], \
           agent_targets_size[s], \
           others_targets_size[s]))
print("\n ========== \n")


for i,r in enumerate(tqdm(roots, leave=True)):
    print("Extracting ", r)
    files = os.listdir(r)
    curr_av, curr_agent, curr_other, trajectories = 0, 0, 0, []
    target_av, target_agent, target_other = av_targets_size[i], agent_targets_size[i], others_targets_size[i]

    for f in tqdm(files, leave=False):
        path = os.path.join(r, f)
        df = pd.read_csv(path)

        # different agents
        agents = df.TRACK_ID.unique().tolist()

        # getting trajectories
        for track_id in agents:
            traj  = df[df.TRACK_ID == track_id].copy()
            object_type = traj.OBJECT_TYPE.iloc[0]
            traj["DATASET_ID"] = int(f[:f.find(".csv")])

            if len(traj) >= full_traj_size:
                # case av
                if object_type == "AV":
                    if curr_av < target_av:
                        curr_av += 1
                        trajectories.append(traj[:full_traj_size])
                # case agent
                elif object_type == "AGENT":
                    if curr_agent < target_agent:
                        curr_agent += 1
                        trajectories.append(traj[:full_traj_size])

                # case others
                elif object_type == "OTHERS":
                    if curr_other < target_other:
                        curr_other += 1
                        trajectories.append(traj[:full_traj_size])

                else:
                    print("Something wrong with the labels in file {}".format(f))
            if curr_av == target_av and curr_agent == target_agent and curr_other == target_other:
                df_total = pd.concat(trajectories).reset_index(drop=True)
                # save df
                df_total.to_csv(os.path.join(save_dir, save_dfs[i]))
                with open(os.path.join(save_dir, save_lists[i]), "wb") as file:
                    cPickle.dump(trajectories, file)
                mode = save_dfs[i][:save_dfs[i].find(".csv")]
                if mode == "train" or mode == "test":
                    break
                else:
                    i += 1
                    curr_av, curr_agent, curr_other, trajectories = 0, 0, 0, []
                    target_av, target_agent, target_other = av_targets_size[i], agent_targets_size[i], others_targets_size[i]
        else:
            continue
        break