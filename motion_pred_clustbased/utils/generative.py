import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import datetime
from .common import *
from ..clustering.utils import save_clustering_object

# from ..clustering.utils import save_clustering_object, FastDiscreteFrechetMatrix, euclidean


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, torch.inf)


def train_vae(
    model: nn.Module,
    x_raw: torch.Tensor,
    y_raw: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    opt,
    cs,
    hyp_cfg,
    labels=None,
    device=torch.device("cpu"),
):
    opt.zero_grad()
    x.requires_grad_()
    y.requires_grad_()
    bs = x.size(0)
    var_info = hyp_cfg["variety"]
    l2_w = hyp_cfg["l2_w"]

    # pass through cvae
    g_l2_loss, loss_traj, loss = [], torch.zeros(1), torch.zeros(1)
    for _ in range(var_info["k_value"]):
        fake_preds, mu, log_var = model(
            x, labels=labels, coord_scaler=cs, x_raw=x_raw, y=y
        )
        out_last = cs.denorm_increment(x_raw, fake_preds.clone())
        if l2_w > 0:
            g_l2_loss.append(
                l2_w
                * F.mse_loss(out_last, y_raw, reduction="none").mean(dim=1).mean(dim=0)
            )

    if l2_w > 0:
        g_l2_loss = torch.stack(g_l2_loss, dim=1)
        loss_tot = torch.min(g_l2_loss, dim=1)[0]
        loss_traj = loss_tot.mean()  # bs
        loss += loss_traj

    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss += hyp_cfg["kl_w"] * kl_loss
    loss.backward()
    opt.step()
    return loss.item()


def train_discriminator(
    disc: nn.Module,
    gen: nn.Module,
    x_raw: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    d_opt,
    clipping_threshold_d=0.0,
    proc_label=None,
    cs=None,
    device=torch.device("cpu"),
):
    disc.zero_grad()
    x.requires_grad_()
    bs = x.size(0)
    ys_real = torch.ones((bs, 1), device=device) * random.uniform(
        0.7, 1.2
    )  # soft labels
    ys_fake = torch.ones((bs, 1), device=device) * random.uniform(0, 0.3)

    # pass through the generator
    with torch.no_grad():
        fake_preds = gen(x, labels=proc_label, coord_scaler=cs, x_raw=x_raw)

    work_as_pred = (
        True if fake_preds.size(1) == y.size(1) else False
    )  # working as a predictor

    # denorm, increment, normalize if concatenated input
    y_disc = fake_preds.detach()
    if disc.inp_dim == 4:
        assert (
            work_as_pred == True
        ), "Working as a generator is not compatible with concatenated input"
        denorm_vect_dir = cs.denorm_coords(fake_preds)
        r = torch.sqrt(denorm_vect_dir[:, :, 0] ** 2 + denorm_vect_dir[:, :, 1] ** 2)
        theta = torch.atan2(denorm_vect_dir[:, :, 1], denorm_vect_dir[:, :, 0])
        polar = torch.stack([r, theta], dim=-1)
        polar_norm = cs.norm_coords(polar, mode="polar")
        y_disc = torch.cat([fake_preds, polar_norm], dim=-1)
    y_disc.requires_grad_()

    # pass fake through the discriminator
    full_fake_traj = concat_subtrajs(x, y_disc) if work_as_pred else fake_preds
    scores_fake = disc(full_fake_traj, labels=proc_label)

    # pass real through the discriminator
    real_traj = concat_subtrajs(x, y)  # seq dimension
    scores_real = disc(real_traj, labels=proc_label)

    # loss
    loss_real = F.binary_cross_entropy_with_logits(scores_real, ys_real)
    loss_fake = F.binary_cross_entropy_with_logits(scores_fake, ys_fake)
    loss = loss_real + loss_fake

    loss.backward()
    if clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(disc.parameters(), clipping_threshold_d)
    d_opt.step()
    return loss.item()


def train_generator(
    disc: nn.Module,
    gen: nn.Module,
    x_raw: torch.Tensor,
    y_raw: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    g_opt,
    cs,
    hyp_cfg,
    labels=None,
    device=torch.device("cpu"),
):
    gen.zero_grad()
    x.requires_grad_()
    bs = x.size(0)
    ys_real = torch.ones((bs, 1), device=device) * random.uniform(
        0.7, 1.2
    )  # soft labels
    var_info = hyp_cfg["variety"]
    l2_w = hyp_cfg["l2_w"]

    # pass through the generator
    g_l2_loss, loss_traj, loss = [], torch.zeros(1), torch.zeros(1)
    for _ in range(var_info["k_value"]):
        fake_preds = gen(x, labels=labels, coord_scaler=cs, x_raw=x_raw)
        work_as_pred = (
            True if fake_preds.size(1) == y_raw.size(1) else False
        )  # working as a predictor
        if work_as_pred:  # predictor out -> pred_len
            out_last = cs.denorm_increment(x_raw, fake_preds.clone())
            y_true = y_raw
        else:  # generator out -> full_traj
            first_ts = x_raw[:, 0, :].unsqueeze(dim=1).clone()
            out_last = cs.denorm_increment(first_ts, fake_preds.clone())
            out_last = torch.cat([first_ts, out_last], dim=1)
            y_true = concat_subtrajs(x_raw, y_raw)
        y_hat = out_last

        if l2_w > 0:
            g_l2_loss.append(
                l2_w
                * F.mse_loss(y_hat, y_true, reduction="none").mean(dim=-1).mean(dim=-1)
            )

    if l2_w > 0:
        g_l2_loss = torch.stack(g_l2_loss, dim=1)
        loss_tot = torch.min(g_l2_loss, dim=1)[0]
        loss_traj = loss_tot.mean()  # bs
        loss += loss_traj

    y_disc = fake_preds.detach()
    if disc.inp_dim == 4:
        denorm_vect_dir = cs.denorm_coords(fake_preds)
        r = torch.sqrt(denorm_vect_dir[:, :, 0] ** 2 + denorm_vect_dir[:, :, 1] ** 2)
        theta = torch.atan2(denorm_vect_dir[:, :, 1], denorm_vect_dir[:, :, 0])
        polar = torch.stack([r, theta], dim=-1)
        polar_norm = cs.norm_coords(polar, mode="polar")
        y_disc = torch.cat([fake_preds, polar_norm], dim=-1)
    y_disc.requires_grad_()

    full_fake_traj = concat_subtrajs(x, y_disc) if work_as_pred else fake_preds

    # Feature matching or van gan
    if hyp_cfg["fm"]:
        feature_real = disc(
            concat_subtrajs(x, y).detach(), get_features=True, labels=labels
        )
        feature_fake = disc(full_fake_traj, get_features=True, labels=labels)
        disc_loss = F.mse_loss(feature_fake, feature_real.detach())
    else:
        scores_fake = disc(full_fake_traj, labels=labels)
        disc_loss = F.binary_cross_entropy_with_logits(scores_fake, ys_real)

    loss += hyp_cfg["adv_w"] * disc_loss
    loss.backward()
    if hyp_cfg["clip_thresh_g"] > 0:
        nn.utils.clip_grad_norm_(gen.parameters(), hyp_cfg["clip_thresh_g"])
    g_opt.step()
    return loss.item()


def create_dirs_vae(cfg, net_cfg, hyp_cfg, inputs_type):
    """Create all necessary folders to store results and outputs for cvae
    Args:
        cfg (dict): config file
        inputs_type (str): type of inputs
    Returns:
        [type]: path for the tensorboard runs, path for the outputs
    """
    ts = str(datetime.datetime.now().strftime("%y-%m-%d_%a_%H:%M:%S:%f")) + str(
        os.getpid()
    )
    model_name = cfg["model"]
    clustering_method = (
        cfg["clustering"]["algorithm"] if model_name == "vae-clust-based" else "None"
    )

    vae_type = net_cfg["net_type"]
    vae_type = vae_type + f"_{net_cfg['state']}" if vae_type == "lstm" else vae_type
    k_var_train, k_var_pred = (
        hyp_cfg["variety"]["k_value"],
        hyp_cfg["variety"]["n_preds"],
    )
    test_name = str(cfg["data"]["dataset_target"])  # test dataset
    model_type = model_name + f"_{vae_type}-{clustering_method}"
    check_dir = cfg["save"]["path"]
    ds = cfg["data"]["dataset"]

    runs_path = os.path.join(
        check_dir,
        "runs",
        ds + f"k-{k_var_train}-{k_var_pred}",
        model_type,
        test_name,
        inputs_type,
    )
    output_path = os.path.join(
        check_dir,
        "outputs",
        ds + f"{k_var_train}-{k_var_pred}",
        model_type,
        test_name,
        inputs_type,
    )  # for clustering, models and so on
    output_path = os.path.join(output_path, ts)
    runs_path = os.path.join(runs_path, ts)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(runs_path):
        os.makedirs(runs_path)
    return runs_path, output_path


def create_dirs(cfg, inputs_type):
    """Create all necessary folders to store results and outputs for gan-based methods
    Args:
        cfg (dict): config file
        inputs_type (str): type of inputs
    Returns:
        [type]: path for the tensorboard runs, path for the outputs
    """
    ts = str(datetime.datetime.now().strftime("%y-%m-%d_%a_%H:%M:%S:%f")) + str(
        os.getpid()
    )
    model_name = cfg["model"]
    # hyperparams
    gen_type = cfg["generator"]["net_type"]
    disc_type = cfg["discriminator"]["net_type"]
    gen_type = (
        gen_type + f"_{cfg['generator']['state']}" if gen_type == "lstm" else gen_type
    )
    disc_type = (
        disc_type + f"_{cfg['discriminator']['state']}"
        if disc_type == "lstm"
        else disc_type
    )
    k_var_train, k_var_pred = (
        cfg["hyperparameters"]["variety"]["k_value"],
        cfg["hyperparameters"]["variety"]["n_preds"],
    )
    loss = "fm" if cfg["hyperparameters"]["fm"] else "adv"
    disc_steps = str(cfg["hyperparameters"]["d_nsteps"])

    test_name = str(cfg["data"]["dataset_target"])  # test dataset
    model_type = model_name + f"_gen-{gen_type}_disc-{disc_type}_nsteps-{disc_steps}"
    check_dir = cfg["save"]["path"]  # dir to save checkpoints, results
    ds = cfg["data"]["dataset"]

    runs_path = os.path.join(
        check_dir,
        "runs",
        ds + f"_loss-{loss}_" + f"k-{k_var_train}-{k_var_pred}",
        model_type,
        test_name,
        inputs_type,
    )
    output_path = os.path.join(
        check_dir,
        "outputs",
        ds + f"_loss-{loss}_" + f"{k_var_train}-{k_var_pred}",
        model_type,
        test_name,
        inputs_type,
    )  # for clustering, models and so on
    if model_name == "sc-gan":  # clustering --> k
        clustering_info = cfg["clustering"]
        clustering = clustering_info["name"]
        runs_path = os.path.join(runs_path, clustering)
        output_path = os.path.join(
            output_path, clustering
        )  # for clustering, models and so on
        cond_name = (
            cfg["generator"]["condition"]["train"]
            + f"-{cfg['generator']['condition']['inference']}"
        )
        runs_path = os.path.join(runs_path, cond_name)
        output_path = os.path.join(output_path, cond_name)
        rec_every = f"recevery_{clustering_info['recluster_every']}"
        k_value = f"k_{clustering_info['kwargs']['n_clusters']}"
        runs_path = os.path.join(
            runs_path, rec_every, k_value
        )  # dir to save tensorboard runs
        output_path = os.path.join(output_path, rec_every, k_value)
        cluster_path = os.path.join(output_path, ts, "clusters")
        if not os.path.exists(cluster_path):  # only for clustering embeddings
            os.makedirs(cluster_path)
    output_path = os.path.join(output_path, ts)
    runs_path = os.path.join(runs_path, ts)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(runs_path):
        os.makedirs(runs_path)
    return runs_path, output_path


class ConditionState:
    def __init__(self, gen_name, coord_scaler, clustering, cond_cfg) -> None:
        self.gen_name = gen_name
        self.cs = coord_scaler
        self.train_cond_type = cond_cfg["train"]  # training condition type
        self.inf_cond_type = cond_cfg["inference"]  # inference condition type
        self.n_samples = cond_cfg["samples"]
        self.n_neighbors = (
            cond_cfg["n_neighbors"] if "n_neighbors" in cond_cfg.keys() else None
        )
        self.k = clustering.k
        self.x = clustering.x
        self.inf_cond_type = (
            "neighbors-ds"
            if clustering.time_series and self.inf_cond_type == "neighbors-fs"
            else self.inf_cond_type
        )

    def __distance_2_centroids(self, clustering, features):
        features_centroids = clustering.method.cluster_centers_[clustering.mapping]
        if clustering.time_series:
            features, features_centroids = features.reshape(
                features.shape[0], -1
            ), features_centroids.reshape(features_centroids.shape[0], -1)
        vect = features[:, None] - torch.from_numpy(features_centroids)
        dists = torch.linalg.norm(vect, dim=-1)
        return dists

    def __train_centroids(self, clustering, full_traj):
        features_traj = clustering.get_features(full_traj)
        dists = self.__distance_2_centroids(clustering, features_traj)
        likelihoods = (1 / (dists + 1e-10)).softmax(dim=-1)
        return likelihoods

    def __inf_silhouette(self, clustering, x, y, x_raw, gen):
        if y is not None:  # debug mode
            gttt = clustering.get_labels(torch.cat([x, y], dim=1), None)
        with torch.no_grad():
            bs, _, inputs_dim = x.shape
            silhouettes = []
            for pp_cl in range(self.k):
                batch_cl = torch.full((bs,), pp_cl)

                avg_silhouette = 0
                for _ in range(self.n_samples):
                    pred = gen(x, labels=batch_cl, coord_scaler=self.cs, x_raw=x_raw)
                    if self.gen_name == "vae-clust-based":
                        pred = pred[0]
                    # prediction
                    if inputs_dim == 2:
                        full_pred = torch.cat([x, pred], dim=1)
                    elif inputs_dim == 4:
                        # denormalize, increment, etc.
                        pred_raw = self.cs.denorm_increment(x_raw, pred)
                        pred_norm = self.cs.norm_coords(pred_raw, mode="raw")
                        preds = torch.cat([pred_norm, pred], dim=-1)
                        full_pred = torch.cat([x, preds], dim=1)
                    features = clustering.get_features(full_pred)
                    # distance to closest centroid of the conditioned cluster (cohesion)
                    dists_cohe = self.__distance_2_centroids(clustering, features)[
                        :, pp_cl
                    ]

                    # distance to closest other cluster's centroid (separation)
                    cent_feat = torch.from_numpy(
                        clustering.method.cluster_centers_[clustering.mapping]
                    )
                    cent_feat = (
                        cent_feat
                        if not clustering.time_series
                        else cent_feat.view(cent_feat.size(0), -1)
                    )
                    dists_cents = pairwise_distances(cent_feat)
                    dists_cents[
                        range(dists_cents.size(0)), range(dists_cents.size(0))
                    ] = torch.inf  # prepare argmin
                    clos_cents = dists_cents.argmin(dim=-1)
                    clos_cents_feat = torch.index_select(
                        cent_feat, 0, clos_cents[batch_cl]
                    )
                    vect_sep = features.reshape(features.shape[0], -1) - clos_cents_feat
                    dists_sep = torch.linalg.norm(vect_sep, dim=-1)
                    avg_silhouette += (dists_sep - dists_cohe) / torch.maximum(
                        dists_cohe, dists_sep
                    )
                silhouettes.append(avg_silhouette / self.n_samples)
            sil = torch.stack(silhouettes, dim=-1)
            likelihoods = (sil + 1e-10).softmax(dim=-1)
        return likelihoods

    def __inf_centroids(self, clustering, x, y, x_raw, gen):
        if y is not None:  # debug mode
            gttt = clustering.get_labels(torch.cat([x, y], dim=1), None)

        with torch.no_grad():
            bs, _, inputs_dim = x.shape
            distances = []
            for pp_cl in range(self.k):
                batch_cl = torch.full((bs,), pp_cl)

                avg_distance = 0
                for _ in range(self.n_samples):
                    pred = gen(x, labels=batch_cl, coord_scaler=self.cs, x_raw=x_raw)
                    if self.gen_name == "vae-clust-based":
                        pred = pred[0]
                    # prediction
                    if inputs_dim == 2:
                        full_pred = torch.cat([x, pred], dim=1)
                    elif inputs_dim == 4:
                        # denormalize, increment, etc.
                        pred_raw = self.cs.denorm_increment(x_raw, pred)
                        pred_norm = self.cs.norm_coords(pred_raw, mode="raw")
                        preds = torch.cat([pred_norm, pred], dim=-1)
                        full_pred = torch.cat([x, preds], dim=1)

                    features = clustering.get_features(full_pred)
                    avg_distance += self.__distance_2_centroids(clustering, features)

                distances.append(avg_distance[:, pp_cl] / self.n_samples)

            distances = torch.stack(distances, dim=-1)
            likelihoods = (1 / (distances + 1e-10)).softmax(dim=-1)
        return likelihoods

    def __inf_kneighbors(self, clustering, x, y, x_raw, gen):
        # fast_frechet = FastDiscreteFrechetMatrix(euclidean)
        with torch.no_grad():
            if y is not None:  # debug mode
                gttt = clustering.get_labels(torch.cat([x, y], dim=1), None)
            bs, _, inputs_dim = x.shape
            # scores
            scores, targ_scores = torch.zeros((self.k, bs)), torch.zeros((self.k, bs))

            # clustering space labels
            labels_pts = clustering.get_labels(self.x, None)
            visited_clusters = []

            features_ds = (
                self.x
                if self.inf_cond_type == "neighbors-ds"
                else torch.from_numpy(clustering.get_cluster_batch_features())
            )
            features_ds = (
                torch.from_numpy(features_ds)
                if isinstance(features_ds, np.ndarray)
                else features_ds
            )

            for pp_cl in range(self.k):
                batch_cl = torch.full((bs,), pp_cl)

                # get pred
                pred = gen(x, labels=batch_cl, coord_scaler=self.cs, x_raw=x_raw)
                if self.gen_name == "vae-clust-based":
                    pred = pred[0]

                # prediction
                if inputs_dim == 2:
                    full_pred = torch.cat([x, pred], dim=1).view(bs, -1)
                if inputs_dim == 4:
                    # denormalize, increment, etc.
                    full_pred_raw = self.cs.denorm_increment(x_raw, pred)
                    pred_traj_raw = torch.cat([x_raw, full_pred_raw], dim=1)
                    pred_traj_norm = self.cs.norm_coords(pred_traj_raw, mode="raw")
                    pred_dir = torch.cat([x[:, :, :2], pred], dim=1)
                    full_pred = torch.cat([pred_traj_norm, pred_dir], dim=-1)

                features_fake = (
                    full_pred
                    if self.inf_cond_type == "neighbors-ds"
                    else clustering.get_features(full_pred)
                )

                scores_intra_cluster = torch.zeros((self.k, bs))
                for cluster in range(self.k):
                    # Going throughout the clustering space to get the k closest neighboors from each cluster
                    targ_idxs = torch.where(labels_pts == cluster)[0]
                    n_trajectories = targ_idxs.size(0)
                    if n_trajectories >= self.n_neighbors:
                        # euclidean distance
                        cmp_features = (
                            features_ds[targ_idxs].view(n_trajectories, -1)
                            if self.inf_cond_type == "neighbors-ds"
                            else features_ds[targ_idxs]
                        )
                        vect = features_fake[:, None] - cmp_features
                        # dist = torch.sum(torch.abs(vect), dim = -1) # L1 norm
                        dist = torch.linalg.norm(vect, dim=-1)  # L2 norm

                        ############################################################
                        # frechet-distance
                        # dist = []
                        # for traj in features_ds[targ_idxs].numpy():
                        #     fd_b = []
                        #     for traj_b in features_fake.numpy():
                        #         fd = fast_frechet.distance(traj_b, traj)
                        #         fd_b.append(fd)
                        #     fd_b = torch.tensor(fd_b)
                        #     dist.append(fd_b)
                        # dist = torch.stack(dist, dim = 0).T

                        sorted_dist = torch.argsort(dist, dim=-1)[:, : self.n_neighbors]
                        dist_2_closest_neig = torch.gather(dist, -1, sorted_dist)
                        avg_dist = (
                            torch.sum(dist_2_closest_neig, dim=-1) / self.n_neighbors
                        )
                        scores_intra_cluster[cluster, :] = avg_dist
                    else:
                        scores_intra_cluster[cluster, :] = 10e6
                    if cluster == pp_cl:
                        targ_scores[pp_cl, :] = scores_intra_cluster[cluster, :]
                visited_clusters.append(scores_intra_cluster)
            idx_min = targ_scores.argmin(dim=0)
            for i in range(bs):
                scores[:, i] = visited_clusters[idx_min[i]][:, i]

            probs = torch.softmax(1 / (scores + 1e-10), axis=0).T
            final_probs = probs.clone()
            final_probs = final_probs.softmax(dim=1)
        return final_probs

    def get_conditions(self, clustering, x, mode, y=None, **kwargs):
        if mode == "train":
            full_traj = torch.cat([x, y], dim=1)
            if self.train_cond_type == "recognizer":
                conds = clustering.get_labels(full_traj, None)
            elif self.train_cond_type == "centroids":
                conds = self.__train_centroids(clustering, full_traj)
            else:
                raise NotImplementedError(f"{self.cond_type}")
        elif mode == "inference":
            if self.inf_cond_type == "recognizer":
                full_traj = torch.cat([x, y], dim=1)
                conds = clustering.get_labels(full_traj, None)
            elif self.inf_cond_type == "centroids":
                conds = self.__inf_centroids(
                    clustering, x, y, kwargs["x_raw"], kwargs["gen"]
                )
            elif self.inf_cond_type == "silhouette":
                conds = self.__inf_silhouette(
                    clustering, x, y, kwargs["x_raw"], kwargs["gen"]
                )
            elif self.inf_cond_type in ["neighbors-ds", "neighbors-fs"]:
                conds = self.__inf_kneighbors(
                    clustering, x, y, kwargs["x_raw"], kwargs["gen"]
                )
            else:
                raise NotImplementedError(f"{self.inf_cond_type}")
        else:
            raise NotImplementedError(f"{mode}")

        return conds


def save_checkpoints(
    epoch: int,
    save_dir: str,
    test_dsname: str,
    model_name: str,
    gen: nn.Module,
    disc: nn.Module,
    g_opt,
    d_opt,
    g_sched,
    d_sched,
    clustering=None,
):
    """_summary_
    Args:
        epoch (int): _description_
        save_dir (str): _description_
        test_dsname (str): _description_
        model_name (str): _description_
        gen (nn.Module): generator and its respective weights
        disc (nn.Module): discriminator and its respective weights
        optimizer (_type_): optimizer
        lr_sched (_type_): learning rate scheduler
    """
    path = os.path.join(
        save_dir, f"checkpoint_{test_dsname}_{model_name}_ep{epoch}.pth"
    )
    checkpoint = {
        "epoch": epoch,
        "gen": gen.state_dict(),
        "disc": disc.state_dict(),
        "g_opt": g_opt.state_dict(),
        "d_opt": d_opt.state_dict(),
        "g_sched": g_sched,
        "d_sched": d_sched,
    }
    torch.save(checkpoint, path)
    if clustering:
        save_clustering_object(save_dir, clustering, epoch)


def save_objects(
    gen: torch.nn.Module, disc: torch.nn.Module, output_dir: str, clustering=None
):
    """Save the outputs of the training
    Args:
        gen (torch.nn.Module): [description]
        disc (torch.nn.Module): [description]
        clusterer ([type], optional): [description]. Defaults to None.
    """

    # saving generator and discriminator
    torch.save(gen.state_dict(), os.path.join(output_dir, "gen.pth"))
    torch.save(disc.state_dict(), os.path.join(output_dir, "disc.pth"))
    # saving clusterer if exists
    if clustering is not None:
        save_clustering_object(output_dir, clustering, "best")


def reparam_trick(mu, log_var):
    std = (0.5 * log_var).exp()
    eps = torch.rand_like(std)
    return mu + eps * std
