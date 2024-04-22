import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from ..utils.common import *


class DetNet(nn.Module):
    def __init__(self, cfg, inp_dim, device):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = cfg["hid_dim"]
        self.emb_dim = cfg["emb_dim"]
        self.device = device
        self.activ = cfg["activ"]
        self.drop = cfg["drop"]
        self.scale = 1.0
        self.net_type = cfg["model"]
        self.pred_len = cfg["pred_len"]
        self.hidden_state = True if cfg["state"] == "hidden" else False

        mlp_dims = cfg["mlp_dims"]
        mlp_dims.insert(0, self.hid_dim)
        mlp_dims.append(cfg["pred_len"] * 2)
        self.encoder = Encoder(
            self.inp_dim,
            self.emb_dim,
            self.hid_dim,
            class_emb_dim=0,
            net_type=self.net_type,
        )
        self.decoder = Decoder(
            self.inp_dim,
            self.emb_dim,
            self.hid_dim,
            2,
            class_emb_dim=0,
            net_type=self.net_type,
        )

    def forward(self, x, **kwargs):
        bs = x.size(0)
        hidden_cell = (
            (
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
            )
            if self.net_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim)
        )

        x = x * self.scale
        # encoder
        inp_lstm = self.encoder.input_embedding(x.contiguous().view(-1, self.inp_dim))
        inp_lstm = inp_lstm.view(bs, -1, self.emb_dim)
        _, hidden_cell_state = self.encoder.temp_feat(inp_lstm, hidden_cell)

        # decoder
        last_pt = x[:, -1, :]
        last_pos, poses = last_pt.view(bs, 1, -1), []
        decoder_input = self.decoder.input_embedding(last_pos)

        for _ in range(self.pred_len):
            _, hidden_cell = self.decoder.temp_feat(decoder_input, hidden_cell_state)
            hn = (
                hidden_cell[0 if self.hidden_state else 1]
                if self.net_type == "lstm"
                else hidden_cell
            )
            hn = hn.view(-1, self.hid_dim)
            decoder_out = self.decoder.predictor(hn)
            pred_pt = decoder_out.view(bs, 1, -1)
            poses.append(pred_pt)
            if self.inp_dim == 4:  # transform to polar coordinates
                denorm_vect_dir = kwargs["coord_scaler"].denorm_coords(pred_pt)
                r = torch.sqrt(
                    denorm_vect_dir[:, :, 0] ** 2 + denorm_vect_dir[:, :, 1] ** 2
                )
                theta = torch.atan2(denorm_vect_dir[:, :, 1], denorm_vect_dir[:, :, 0])
                polar = torch.stack([r, theta], dim=-1)
                polar_norm = kwargs["coord_scaler"].norm_coords(polar, mode="polar")
                y_gen = torch.cat([pred_pt, polar_norm], dim=-1)
            else:
                y_gen = pred_pt
            decoder_input = self.decoder.input_embedding(y_gen)
        pred_traj = torch.cat(poses, dim=1)
        return pred_traj


class ANet(nn.Module):
    #     """ Auxiliar network object
    #     """
    def __init__(self, cfg: dict, n_clusters: int, device) -> None:
        super().__init__()
        self.mode = cfg["mode"]
        self.net_type = cfg["net_type"]

        self.inp_dim = cfg["inp_features"]
        self.inp_seq_len = cfg["inp_seq_len"]
        self.pred_len = cfg["pred_len"]

        self.n_layers = cfg["n_layers"]
        self.bidir = cfg["bidirectional"]
        self.emb_dim = cfg["emb_dim"]
        self.hid_dim = cfg["hid_dim"]
        self.bn = cfg["batchnorm"]
        self.activ = cfg["activ"]
        self.drop = cfg["drop"]
        self.n_samples = cfg["n_samples"]

        self.n_clusters = n_clusters
        self.scale = 1.0
        self.device = device

        self.inp_emb = (
            Encoder(
                self.inp_dim, self.emb_dim, self.hid_dim, class_emb_dim=0, net_type=None
            )
            if self.mode in ["multiclass", "multiclass_gen"]
            else [
                Encoder(
                    self.inp_dim,
                    self.emb_dim,
                    self.hid_dim,
                    class_emb_dim=0,
                    net_type=None,
                )
                for _ in range(self.n_clusters)
            ]
        )

        if self.net_type == "lstm":
            self.feat_ext = (
                nn.LSTM(
                    input_size=self.emb_dim,
                    hidden_size=self.hid_dim,
                    num_layers=self.n_layers,
                    batch_first=True,
                    dropout=self.drop if self.n_layers > 1 else 0,
                    bidirectional=self.bidir,
                )
                if self.mode in ["multiclass", "multiclass_gen"]
                else [
                    nn.LSTM(
                        input_size=self.emb_dim,
                        hidden_size=self.hid_dim,
                        num_layers=self.n_layers,
                        batch_first=True,
                        dropout=self.drop if self.n_layers > 1 else 0,
                        bidirectional=self.bidir,
                    )
                    for _ in range(self.n_clusters)
                ]
            )
        elif self.net_type == "gru":
            self.feat_ext = (
                nn.GRU(
                    input_size=self.emb_dim,
                    hidden_size=self.hid_dim,
                    num_layers=self.n_layers,
                    batch_first=True,
                    dropout=self.drop if self.n_layers > 1 else 0,
                    bidirectional=self.bidir,
                )
                if self.mode in ["multiclass", "multiclass_gen"]
                else [
                    nn.GRU(
                        input_size=self.emb_dim,
                        hidden_size=self.hid_dim,
                        num_layers=self.n_layers,
                        batch_first=True,
                        dropout=self.drop if self.n_layers > 1 else 0,
                        bidirectional=self.bidir,
                    )
                    for _ in range(self.n_clusters)
                ]
            )
        elif self.net_type == "ff":
            self.feat_ext = (
                nn.Sequential(
                    nn.Linear(self.emb_dim * self.inp_seq_len, self.hid_dim),
                    nn.PReLU(),
                    nn.Dropout(self.drop),
                )
                if self.mode in ["multiclass", "multiclass_gen"]
                else [
                    nn.Sequential(
                        nn.Linear(self.emb_dim * self.inp_seq_len, self.hid_dim),
                        nn.PReLU(),
                        nn.Dropout(self.drop),
                    )
                    for _ in range(self.n_clusters)
                ]
            )
        else:
            raise NotImplementedError(f"{self.net_type}")

        classifier_dims = (
            [self.hid_dim, int(self.hid_dim / 2)]
            if (not self.bidir) or self.net_type == "ff"
            else [self.hid_dim * 2, int(self.hid_dim), int(self.hid_dim / 2)]
        )
        classifier_dims[0] = (
            int(classifier_dims[0] * (self.n_clusters))
            if self.mode == "multiclass_concat"
            else classifier_dims[0]
        )

        self.classifier = make_mlp(
            classifier_dims,
            activation=self.activ,
            batch_norm=self.bn,
            dropout=self.drop,
        )
        self.fc_out = nn.Linear(int(self.hid_dim / 2), n_clusters)
        self.last_activ = nn.Softmax(dim=-1)

    def get_inference_out(
        self, x: torch.Tensor, gen: nn.Module, gt: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            self.eval()

            if self.mode in ["multiclass", "multiclass_gen"]:
                bs, likelihoods = x.size(0), torch.zeros(
                    x.size(0), self.n_clusters, self.n_clusters
                )
                for i in range(self.n_clusters):
                    pp_cl = torch.full((bs,), i)
                    for _ in range(self.n_samples):
                        fake_preds = gen(x, labels=pp_cl)
                        if isinstance(fake_preds, tuple):
                            fake_preds = fake_preds[0]
                        full_pred_traj = concat_subtrajs(x, fake_preds)
                        target_traj = (
                            full_pred_traj
                            if self.inp_seq_len == full_pred_traj.size(1)
                            else fake_preds
                        )
                        likelihoods[:, i, :] += self.forward(target_traj)
                idx_target = likelihoods.diagonal(dim1=1, dim2=2).argmax(dim=-1)
                idx_target = idx_target.unsqueeze(dim=-1).unsqueeze(dim=-1)
                idx_target = torch.repeat_interleave(idx_target, self.n_clusters, dim=2)
                out = torch.gather(likelihoods, 1, idx_target).squeeze()
            else:
                bs, likelihoods = x.size(0), torch.zeros(x.size(0), self.n_clusters)
                for _ in range(self.n_samples):
                    trajs_2_eval = []
                    for i in range(self.n_clusters):
                        pp_cl = torch.full((bs,), i)
                        fake_preds = gen(x, labels=pp_cl)
                        if isinstance(fake_preds, tuple):
                            fake_preds = fake_preds[0]
                        full_pred_traj = concat_subtrajs(x, fake_preds)
                        target_traj = (
                            full_pred_traj
                            if self.inp_seq_len == full_pred_traj.size(1)
                            else fake_preds
                        )
                        trajs_2_eval.append(target_traj)
                    trajs_2_eval = torch.stack(trajs_2_eval, dim=1)
                    likelihoods += self.forward(trajs_2_eval)
                out = likelihoods
        out = out.softmax(dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.size(1) == self.inp_seq_len or x.size(2) == self.inp_seq_len
        ), f"The input shape does not match the cfg file -> {x.size(1)}!={self.inp_seq_len}"

        bs = x.size(0)
        x = x * self.scale

        if self.mode in ["multiclass", "multiclass_gen"]:
            hidden_cell = None
            if self.net_type in ["lstm", "gru"]:
                hidden_cell = (
                    (
                        (
                            torch.zeros(
                                self.n_layers * 2 if self.bidir else self.n_layers,
                                bs,
                                self.hid_dim,
                            ).to(self.device)
                        ),
                        (
                            torch.zeros(
                                self.n_layers * 2 if self.bidir else self.n_layers,
                                bs,
                                self.hid_dim,
                            ).to(self.device)
                        ),
                    )
                    if self.net_type == "lstm"
                    else torch.zeros(
                        self.n_layers * 2 if self.bidir else self.n_layers,
                        bs,
                        self.hid_dim,
                    ).to(self.device)
                )

            inp_feat = self.inp_emb.input_embedding(
                x.contiguous().view(-1, self.inp_dim)
            )
            inp_feat = inp_feat.view(bs, -1, self.emb_dim)

            if self.net_type in ["lstm", "gru"]:
                output, _ = self.feat_ext(inp_feat, hidden_cell)
                features = output[:, -1, :]
            else:
                features = self.feat_ext(inp_feat.view(bs, -1))
        elif self.mode == "multiclass_concat":
            assert (
                x.size(1) == self.n_clusters
            ), f"The cluster dimension {x.size(1)} does not match the cfg file {self.n_clusters}"
            features = []
            for n_cl in range(self.n_clusters):
                if self.net_type in ["lstm", "gru"]:
                    hidden_cell = (
                        (
                            (
                                torch.zeros(
                                    self.n_layers * 2 if self.bidir else self.n_layers,
                                    bs,
                                    self.hid_dim,
                                ).to(self.device)
                            ),
                            (
                                torch.zeros(
                                    self.n_layers * 2 if self.bidir else self.n_layers,
                                    bs,
                                    self.hid_dim,
                                ).to(self.device)
                            ),
                        )
                        if self.net_type == "lstm"
                        else torch.zeros(
                            self.n_layers * 2 if self.bidir else self.n_layers,
                            bs,
                            self.hid_dim,
                        ).to(self.device)
                    )
                inp = x[:, n_cl, :, :]
                inp_feat = self.inp_emb[n_cl].input_embedding(
                    inp.contiguous().view(-1, self.inp_dim)
                )
                inp_feat = inp_feat.view(bs, -1, self.emb_dim)
                if self.net_type in ["lstm", "gru"]:
                    output, _ = self.feat_ext[n_cl](inp_feat, hidden_cell)
                    ind_features = output[:, -1, :]
                else:
                    ind_features = self.feat_ext[n_cl](inp_feat.view(bs, -1))
                features.append(ind_features)
            features = torch.cat(features, dim=-1)

        out = self.classifier(features)
        out = self.last_activ(self.fc_out(out))
        return out


if __name__ == "__main__":
    cfg = {
        "mode": "multiclass_gen",
        "net_type": "gru",
        "n_layers": 2,
        "bidirectional": False,
        "batchnorm": True,
        "emb_dim": 16,
        "hid_dim": 64,
        "activ": "prelu",
        "drop": 0.2,
        "n_samples": 20,
        "pred_len": 12,
        "inp_seq_len": 12,
        "inp_features": 2,
    }
    n_clusters = 13
    bs = 64
    n_trajs = 20
    seq_len = 12
    inp_dim = 2
    model = ANet(cfg, n_clusters, torch.device("cpu"))
    input = torch.rand(bs, n_trajs, seq_len, inp_dim)
    out = model(input)

    assert (
        out.shape[-1] == n_clusters
    ), f"Something wrong with the output shape {out.shape}"
    print("ANet tested")
    print(model)
