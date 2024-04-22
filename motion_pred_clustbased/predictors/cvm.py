import numpy as np
from tqdm import tqdm
from scipy.signal.windows import gaussian


class Predictor_CVM:
    def __init__(self, cfg) -> None:
        self.v0_mode = cfg["param"]["meta"]["v0_mode"]
        self.v0_sigma = cfg["param"]["meta"]["v0_sigma"]
        self.pred_len = cfg["data"]["pred_len"]

    def predict_dataset(self, dataset):
        prediction, position, v_mean = [], [], []
        for n_t, trajectory in tqdm(enumerate(dataset)):
            print(f"Predicting trajectory # {n_t}")
            last_position = trajectory[-1, :]
            mean_velocity = self.average_velocity(trajectory)
            position.append(last_position)
            v_mean.append(mean_velocity)
        v_mean = np.squeeze(v_mean)
        last_position = np.array(position)
        for _ in range(self.pred_len):
            new_position = last_position + v_mean
            prediction.append(new_position)
            last_position = new_position
        return np.stack(prediction, axis=1)

    def average_velocity(self, trajectory):
        velocity = (trajectory[1::, :] - trajectory[:-1, :])
        weights = np.expand_dims(self.get_w(velocity), axis=0)
        new_velocity = np.dot(weights, velocity)
        return new_velocity

    def get_w(self, velocity):
        velocity_len = len(velocity)
        if self.v0_mode == "linear":
            weights = np.ones(velocity_len) / velocity_len
        elif self.v0_mode == "gaussian":
            window = gaussian(2 * velocity_len, self.v0_sigma)
            w1 = window[0:velocity_len]
            scale = np.sum(w1)
            # scale = np.linalg.norm(w1)
            weights = w1 / scale
            # print(w)
        elif self.v0_mode == "constant":
            weights = np.zeros(velocity_len)
            weights[-1] = 1
        else:
            raise NotImplementedError(self.v0_mode)
        return weights


if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from ..utils.common import load_config
    import _pickle as cPickle
    from ..eval.common import FDE, MeanSquaredError
    import torch

    parser = ArgumentParser(
        description="Train deterministic models for trajectory prediction"
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="motion_pred_clustbased/cfg/predict/argoverse/cvm.yaml",
        required=False,
        help="configuration file comprising: networks design choices, hyperparameters, etc.",
    )

    args = parser.parse_args()
    cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/predict/default.yaml")
    preditor = Predictor_CVM(cfg)
    obs_len = cfg["data"]["obs_len"]
    fp = os.path.join(
        cfg["data"]["data_dir"],
        str(cfg["data"]["dataset_target"]),
        f"pp_skip{str(cfg['data']['skip'])}_minped{str(cfg['data']['min_ped'])}",
        "test_raw.pkl",
    )

    # metrics
    ade = MeanSquaredError(squared=False)
    fde = FDE()

    # data
    with open(fp, "rb") as f:
        data = cPickle.load(f)
    dataset = np.stack(data)
    y_hat = preditor.predict_dataset(dataset[:, :obs_len, :])
    y_true = dataset[:, obs_len:, :]

    # compute metrics
    ade.update(preds=torch.from_numpy(y_hat), target=torch.from_numpy(y_true))
    fde.update(preds=torch.from_numpy(y_hat), target=torch.from_numpy(y_true))
    ade_res = ade.compute().item()
    fde_res = fde.compute().item()
    print("\nMetrics:")
    print(f"ADE={ade_res}, FDE={fde_res}")
    ade.reset()
    fde.reset()
