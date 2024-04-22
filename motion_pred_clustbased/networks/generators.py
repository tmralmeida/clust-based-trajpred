from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from ..utils.generative import cat_class_emb, reparam_trick


class VanillaGenerator(nn.Module):
    """Vanilla generator object"""

    def __init__(self, cfg, inp_dim, device):
        super().__init__()
        self.device = device
        self.inp_dim = inp_dim
        self.emb_dim = cfg["emb_dim"]
        self.hid_dim = cfg["hid_dim"]
        self.out_dim = cfg["pred_len"]
        self.noise_dim = cfg["noise_dim"]
        self.noise_type = cfg["noise_type"]
        self.activ = cfg["activ"]
        self.drop = cfg["drop"]
        self.net_type = cfg["net_type"]
        self.hidden_state = True if cfg["state"] == "hidden" else False
        self.scale = 1.0

        # encoder
        if self.inp_dim > 0:
            self.encoder = Encoder(
                self.inp_dim,
                self.emb_dim,
                self.hid_dim,
                class_emb_dim=0,
                net_type=self.net_type,
            )

            # encoder-decoder connection
            mlp_decoder_context_dims = [
                self.hid_dim,
                self.emb_dim,
                self.hid_dim - self.noise_dim,
            ]
            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims, activation=self.activ, dropout=self.drop
            )

            # decoder
            self.decoder = Decoder(
                self.inp_dim,
                self.emb_dim,
                self.hid_dim,
                2,
                class_emb_dim=0,
                net_type=self.net_type,
            )

    def adding_noise(self, hidden_cell: torch.Tensor):
        hidden_state, cell_state = (
            hidden_cell if self.net_type == "lstm" else (hidden_cell, None)
        )

        new_hidden_state = self.mlp_decoder_context(hidden_state)
        z = get_noise((self.noise_dim,), self.noise_type, self.device)
        z_decoder = z.repeat(1, new_hidden_state.size(1), 1)
        new_hidden_state = torch.cat([new_hidden_state, z_decoder], dim=-1)

        if cell_state is not None:  # lstm
            return ((new_hidden_state), (cell_state))
        else:
            return new_hidden_state

    def forward(self, x, **kwargs):
        bs = x.size(0)
        hidden_cell = (
            (
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
            )
            if self.net_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim).to(self.device)
        )

        x = x * self.scale
        # encoder
        inp_lstm = self.encoder.input_embedding(x.contiguous().view(-1, self.inp_dim))
        inp_lstm = inp_lstm.view(bs, -1, self.emb_dim)
        _, enc_hidden_cell = self.encoder.temp_feat(inp_lstm, hidden_cell)
        hidden_cell_state = self.adding_noise(enc_hidden_cell)

        # decoder
        last_pt = x[:, -1, :]
        last_pos, poses = last_pt.view(bs, 1, -1), []
        decoder_input = self.decoder.input_embedding(last_pos)

        for _ in range(self.out_dim):
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


class VanillaVAE(VanillaGenerator):
    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)
        self.z_mu = nn.Linear(self.hid_dim, self.noise_dim)
        self.z_var = nn.Linear(self.hid_dim, self.noise_dim)
        self.encoder_kl = Encoder(
            self.inp_dim,
            self.emb_dim,
            self.hid_dim,
            class_emb_dim=0,
            net_type=self.net_type,
        )

    def sample_z(
        self,
        batch_size: int,
        hidden_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_state is None:  # inference
            return torch.randn(1, batch_size, self.noise_dim), None, None
        hidden_state = hidden_state.squeeze()
        mean_, log_var = self.z_mu(hidden_state), self.z_var(hidden_state)

        # reparameterization trick
        random_sample = reparam_trick(mean_, log_var).unsqueeze(dim=0)
        return random_sample, mean_, log_var

    def forward(self, x, **kwargs):
        training = "y" in kwargs
        bs = x.size(0)
        hidden_cell = (
            (
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
            )
            if self.net_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim).to(self.device)
        )

        x = x * self.scale
        x_inp_lstm = self.encoder.input_embedding(x.contiguous().view(-1, self.inp_dim))
        x_inp_lstm = x_inp_lstm.view(bs, -1, self.emb_dim)
        _, x_enc_hidden_cell = self.encoder.temp_feat(x_inp_lstm, hidden_cell)
        x_hidden_state, x_cell_state = (
            x_enc_hidden_cell if self.net_type == "lstm" else (x_enc_hidden_cell, None)
        )

        y_hidden_state = None
        if training:
            y = kwargs["y"]
            y = y * self.scale
            bs = y.size(0)
            y_inp_lstm = self.encoder_kl.input_embedding(
                y.contiguous().view(-1, self.inp_dim)
            )
            y_inp_lstm = y_inp_lstm.view(bs, -1, self.emb_dim)
            _, y_enc_hidden_cell = self.encoder_kl.temp_feat(y_inp_lstm, hidden_cell)
            y_hidden_state, _ = (
                y_enc_hidden_cell
                if self.net_type == "lstm"
                else (y_enc_hidden_cell, None)
            )

        # random noise
        random_noise, mean_, log_var = self.sample_z(bs, y_hidden_state)
        # concat noise with ouput of observation encoder
        new_hidden_state = self.mlp_decoder_context(x_hidden_state)
        new_hidden_state = torch.cat([new_hidden_state, random_noise], dim=-1)
        new_hidden_cell = (
            ((new_hidden_state), (x_cell_state))
            if x_cell_state is not None
            else new_hidden_state
        )

        # decoder
        last_pt = x[:, -1, :]
        last_pos, poses = last_pt.view(bs, 1, -1), []
        decoder_input = self.decoder.input_embedding(last_pos)
        for _ in range(self.out_dim):
            _, hidden_cell = self.decoder.temp_feat(decoder_input, new_hidden_cell)
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
        return pred_traj, mean_, log_var


class SupCondGenerator(VanillaGenerator):
    """Generator conditioned on supervised classes"""

    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)
        self.k_value = cfg["n_labels"]
        self.cond_type = cfg["condition_type"]
        self.emb_layer = (
            LatentEmbedding(self.k_value, self.k_value)
            if self.cond_type == "embedding"
            else None
        )

        self.encoder = Encoder(
            self.inp_dim + self.k_value,
            self.emb_dim,
            self.hid_dim,
            class_emb_dim=0,
            net_type=self.net_type,
        )

        # decoder
        self.decoder = Decoder(
            self.inp_dim,
            self.emb_dim,
            self.hid_dim,
            2,
            class_emb_dim=self.k_value,
            net_type=self.net_type,
        )

        # encoder-decoder connection
        mlp_decoder_context_dims = [self.hid_dim, self.hid_dim - self.noise_dim]
        self.mlp_decoder_context = make_mlp(
            mlp_decoder_context_dims, [self.activ], dropout=self.drop
        )

    def forward(self, x, **kwargs):
        bs = x.size(0)
        hidden_cell = (
            (
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
            )
            if self.net_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim).to(self.device)
        )

        x = x * self.scale
        labels = (
            self.emb_layer(kwargs["labels"])
            if self.cond_type == "embedding"
            else F.one_hot(kwargs["labels"], num_classes=self.k_value).float()
        )
        # encoder
        inp_emb = cat_class_emb(x, labels)
        obs_emb = self.encoder.input_embedding(
            inp_emb.contiguous().view(-1, self.inp_dim + self.k_value)
        )
        obs_emb = obs_emb.view(bs, -1, self.emb_dim)
        _, enc_hidden_cell = self.encoder.temp_feat(obs_emb, hidden_cell)
        hidden_cell_state = self.adding_noise(enc_hidden_cell)

        # decoder
        last_pt = x[:, -1, :]
        last_pos, poses = last_pt.view(bs, 1, -1), []
        decoder_input = cat_class_emb(last_pos, labels)
        decoder_input = self.decoder.input_embedding(decoder_input)
        for _ in range(self.out_dim):
            _, hidden_cell = self.decoder.temp_feat(decoder_input, hidden_cell_state)
            hn = (
                hidden_cell[0 if self.hidden_state else 1]
                if self.net_type == "lstm"
                else hidden_cell
            )
            hn = hn.view(-1, self.hid_dim)
            decoder_out = self.decoder.predictor(hn)
            pred_pt = decoder_out.view(bs, 1, 2)
            poses.append(pred_pt)

            y_gen = pred_pt
            decoder_input = cat_class_emb(y_gen, labels)
            decoder_input = self.decoder.input_embedding(decoder_input)
        pred_traj = torch.cat(poses, dim=1)
        return pred_traj


class SupCondVAE(SupCondGenerator):
    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)
        self.z_mu = nn.Linear(self.hid_dim, self.noise_dim)
        self.z_var = nn.Linear(self.hid_dim, self.noise_dim)
        self.encoder_kl = Encoder(
            self.inp_dim + self.k_value,
            self.emb_dim,
            self.hid_dim,
            class_emb_dim=0,
            net_type=self.net_type,
        )

    def sample_z(
        self,
        batch_size: int,
        hidden_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_state is None:  # inference
            return torch.randn(1, batch_size, self.noise_dim), None, None
        hidden_state = hidden_state.squeeze()
        mean_, log_var = self.z_mu(hidden_state), self.z_var(hidden_state)

        # reparameterization trick
        random_sample = reparam_trick(mean_, log_var).unsqueeze(dim=0)
        return random_sample, mean_, log_var

    def forward(self, x, **kwargs):
        training = "y" in kwargs
        bs = x.size(0)
        hidden_cell = (
            (
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
            )
            if self.net_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim).to(self.device)
        )

        x = x * self.scale
        labels = (
            self.emb_layer(kwargs["labels"])
            if self.cond_type == "embedding"
            else F.one_hot(kwargs["labels"], num_classes=self.k_value).float()
        )
        # encoder
        x_inp_emb = cat_class_emb(x, labels)
        obs_emb = self.encoder.input_embedding(
            x_inp_emb.contiguous().view(-1, self.inp_dim + self.k_value)
        )
        obs_emb = obs_emb.view(bs, -1, self.emb_dim)
        _, x_enc_hidden_cell = self.encoder.temp_feat(obs_emb, hidden_cell)
        x_hidden_state, x_cell_state = (
            x_enc_hidden_cell if self.net_type == "lstm" else (x_enc_hidden_cell, None)
        )

        y_hidden_state = None
        if training:
            y = kwargs["y"]
            y = y * self.scale
            bs = y.size(0)
            y_inp_emb = cat_class_emb(x, labels)
            y_inp_lstm = self.encoder_kl.input_embedding(
                y_inp_emb.contiguous().view(-1, self.inp_dim + self.k_value)
            )
            y_inp_lstm = y_inp_lstm.view(bs, -1, self.emb_dim)
            _, y_enc_hidden_cell = self.encoder_kl.temp_feat(y_inp_lstm, hidden_cell)
            y_hidden_state, _ = (
                y_enc_hidden_cell
                if self.net_type == "lstm"
                else (y_enc_hidden_cell, None)
            )

        # random noise
        random_noise, mean_, log_var = self.sample_z(bs, y_hidden_state)
        # concat noise with ouput of observation encoder
        new_hidden_state = self.mlp_decoder_context(x_hidden_state)
        new_hidden_state = torch.cat([new_hidden_state, random_noise], dim=-1)
        new_hidden_cell = (
            ((new_hidden_state), (x_cell_state))
            if x_cell_state is not None
            else new_hidden_state
        )

        # decoder
        last_pt = x[:, -1, :]
        last_pos, poses = last_pt.view(bs, 1, -1), []
        decoder_input = cat_class_emb(last_pos, labels)
        decoder_input = self.decoder.input_embedding(decoder_input)
        for _ in range(self.out_dim):
            _, hidden_cell = self.decoder.temp_feat(decoder_input, new_hidden_cell)
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
            decoder_input = cat_class_emb(y_gen, labels)
            decoder_input = self.decoder.input_embedding(decoder_input)
        pred_traj = torch.cat(poses, dim=1)
        return pred_traj, mean_, log_var


class SoftCondGenerator(SupCondGenerator):
    """Generator conditioned on unsupervised signals"""

    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)

    def forward(self, x, **kwargs):
        bs = x.size(0)
        hidden_cell = (
            (
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
            )
            if self.net_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim).to(self.device)
        )

        x = x * self.scale
        # encoder
        if self.cond_type == "embedding":
            labels = (
                kwargs["labels"].argmax(dim=-1).long()
                if kwargs["labels"].dim() > 1
                else kwargs["labels"].long()
            )
            labels = self.emb_layer(labels)
        elif self.cond_type == "one_hot":
            labels = (
                kwargs["labels"].argmax(dim=-1).long()
                if kwargs["labels"].dim() > 1
                else kwargs["labels"].long()
            )
            labels = F.one_hot(labels, num_classes=self.k_value).float()
        inp_emb = cat_class_emb(x, labels)
        obs_emb = self.encoder.input_embedding(
            inp_emb.contiguous().view(-1, self.inp_dim + self.k_value)
        )
        obs_emb = obs_emb.view(bs, -1, self.emb_dim)
        _, enc_hidden_cell = self.encoder.temp_feat(obs_emb, hidden_cell)
        hidden_cell_state = self.adding_noise(enc_hidden_cell)

        # decoder
        last_pt = x[:, -1, :]
        last_pos, poses = last_pt.view(bs, 1, -1), []
        decoder_input = cat_class_emb(last_pos, labels)
        decoder_input = self.decoder.input_embedding(decoder_input)
        for _ in range(self.out_dim):
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
            decoder_input = cat_class_emb(y_gen, labels)
            decoder_input = self.decoder.input_embedding(decoder_input)
        pred_traj = torch.cat(poses, dim=1)
        return pred_traj


class FTSoftCondGenerator(nn.Module):
    """Generator conditioned on unsupervised labels for full trajectories"""

    def __init__(self, cfg, device):
        super().__init__()
        self.pred_len = cfg["pred_len"]
        self.noise_type = cfg["noise_type"]
        self.hid_dim = cfg["hid_dim"]
        self.k_value = cfg["n_labels"]
        self.noise_dim = cfg["noise_dim"]
        self.activ = cfg["activ"]
        self.drop = cfg["drop"]
        self.cond_type = cfg["condition_type"]
        self.device = device

        self.emb_layer = (
            LatentEmbedding(self.k_value, self.k_value)
            if self.cond_type == "embedding"
            else None
        )
        mlp_decoder = [self.noise_dim + self.k_value, self.hid_dim]
        self.mlp_decoder = make_mlp(mlp_decoder, [self.activ], dropout=self.drop)
        self.predictor = nn.Linear(self.hid_dim, self.pred_len * 2)

    def forward(self, x, **kwargs):
        bs = x.size(0)
        z = get_noise(
            (
                bs,
                self.noise_dim,
            ),
            self.noise_type,
            self.device,
        )
        if self.cond_type == "embedding":
            labels = (
                kwargs["labels"].argmax(dim=-1).long()
                if kwargs["labels"].dim() > 1
                else kwargs["labels"].long()
            )
            labels = self.emb_layer(labels)
        elif self.cond_type == "one_hot":
            labels = (
                kwargs["labels"].argmax(dim=-1).long()
                if kwargs["labels"].dim() > 1
                else kwargs["labels"].long()
            )
            labels = F.one_hot(labels, num_classes=self.k_value).float()
        decoder_input = cat_class_emb(z, labels)
        decoder_output = self.mlp_decoder(decoder_input)
        pred_traj = self.predictor(decoder_output)
        return pred_traj.view(bs, -1, 2)


class FTGenerator(VanillaGenerator):
    """Unconditioned generator of full trajectories"""

    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)
        mlp_decoder = [self.noise_dim, self.hid_dim]
        self.mlp_decoder = make_mlp(mlp_decoder, [self.activ], dropout=self.drop)
        self.predictor = nn.Linear(self.hid_dim, self.out_dim * 2)

    def forward(self, x, **kwargs):
        bs = x.size(0)
        z = get_noise(
            (
                bs,
                self.noise_dim,
            ),
            self.noise_type,
            self.device,
        )
        decoder_output = self.mlp_decoder(z)
        pred_traj = self.predictor(decoder_output)
        return pred_traj.view(bs, -1, 2)


class cVAETraj(SoftCondGenerator):
    """conditional variational autoencoder for trajectory prediction"""

    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)
        self.z_mu = nn.Linear(self.hid_dim, self.noise_dim)
        self.z_var = nn.Linear(self.hid_dim, self.noise_dim)
        self.encoder_kl = Encoder(
            self.inp_dim + self.k_value,
            self.emb_dim,
            self.hid_dim,
            class_emb_dim=0,
            net_type=self.net_type,
        )

    def sample_z(
        self,
        batch_size: int,
        hidden_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_state is None:  # inference
            return torch.randn(1, batch_size, self.noise_dim), None, None
        hidden_state = hidden_state.squeeze()
        mean_, log_var = self.z_mu(hidden_state), self.z_var(hidden_state)

        # reparameterization trick
        random_sample = reparam_trick(mean_, log_var).unsqueeze(dim=0)
        return random_sample, mean_, log_var

    def forward(self, x, **kwargs):
        if self.cond_type == "embedding":
            labels = (
                kwargs["labels"].argmax(dim=-1).long()
                if kwargs["labels"].dim() > 1
                else kwargs["labels"].long()
            )
            labels = self.emb_layer(labels)
        elif self.cond_type == "one_hot":
            labels = (
                kwargs["labels"].argmax(dim=-1).long()
                if kwargs["labels"].dim() > 1
                else kwargs["labels"].long()
            )
            labels = F.one_hot(labels, num_classes=self.k_value).float()
        else:
            raise NotImplementedError(self.cond_type)

        training = "y" in kwargs
        bs = x.size(0)
        hidden_cell = (
            (
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
                (torch.zeros(1, bs, self.hid_dim).to(self.device)),
            )
            if self.net_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim).to(self.device)
        )

        x = x * self.scale
        # encoder
        x_inp_emb = cat_class_emb(x, labels)
        obs_emb = self.encoder.input_embedding(
            x_inp_emb.contiguous().view(-1, self.inp_dim + self.k_value)
        )
        obs_emb = obs_emb.view(bs, -1, self.emb_dim)
        _, x_enc_hidden_cell = self.encoder.temp_feat(obs_emb, hidden_cell)
        x_hidden_state, x_cell_state = (
            x_enc_hidden_cell if self.net_type == "lstm" else (x_enc_hidden_cell, None)
        )

        y_hidden_state = None
        if training:
            y = kwargs["y"]
            y = y * self.scale
            bs = y.size(0)
            y_inp_emb = cat_class_emb(x, labels)
            y_inp_lstm = self.encoder_kl.input_embedding(
                y_inp_emb.contiguous().view(-1, self.inp_dim + self.k_value)
            )
            y_inp_lstm = y_inp_lstm.view(bs, -1, self.emb_dim)
            _, y_enc_hidden_cell = self.encoder_kl.temp_feat(y_inp_lstm, hidden_cell)
            y_hidden_state, _ = (
                y_enc_hidden_cell
                if self.net_type == "lstm"
                else (y_enc_hidden_cell, None)
            )

        # random noise
        random_noise, mean_, log_var = self.sample_z(bs, y_hidden_state)
        # concat noise with ouput of observation encoder
        new_hidden_state = self.mlp_decoder_context(x_hidden_state)
        new_hidden_state = torch.cat([new_hidden_state, random_noise], dim=-1)
        new_hidden_cell = (
            ((new_hidden_state), (x_cell_state))
            if x_cell_state is not None
            else new_hidden_state
        )

        # decoder
        last_pt = x[:, -1, :]
        last_pos, poses = last_pt.view(bs, 1, -1), []
        decoder_input = cat_class_emb(last_pos, labels)
        decoder_input = self.decoder.input_embedding(decoder_input)
        for _ in range(self.out_dim):
            _, hidden_cell = self.decoder.temp_feat(decoder_input, new_hidden_cell)
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
            decoder_input = cat_class_emb(y_gen, labels)
            decoder_input = self.decoder.input_embedding(decoder_input)
        pred_traj = torch.cat(poses, dim=1)
        return pred_traj, mean_, log_var


if __name__ == "__main__":
    from argparse import ArgumentParser
    from ..utils.common import *

    parser = ArgumentParser(description="Train DL-based forecasters on 2D motion data")

    parser.add_argument(
        "--cfg",
        type=str,
        default="motion_pred_clustbased/cfg/training/thor/clust_based.yaml",
        required=False,
        help="configuration file comprising: networks design choices, hyperparameters, etc.",
    )

    args = parser.parse_args()
    cfg = load_config(args.cfg, "motion_pred_clustbased/cfg/training/default.yaml")
    model_name = cfg["model"]

    cfg_cvae = cfg["cvae"]
    bs, obs_len, pred_len, inp_dim = 64, 8, 12, 2
    model = cVAETraj(cfg_cvae, inp_dim, torch.device("cpu"))
    input_traj = torch.rand(bs, obs_len, inp_dim)
    classes = torch.randint(0, 11, (bs,))
    assert model(input_traj, labels=classes)[0].shape == (
        bs,
        pred_len,
        inp_dim,
    ), "Something wrong with cvae out shape"
