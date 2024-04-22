import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *


class FFVanDiscriminator(nn.Module):
    """ Feed forward Vanilla network object
    """
    def __init__(self, cfg, inp_dim, device):
        super().__init__()
        self.device = device
        self.inp_dim = inp_dim
        self.pred_len = cfg["pred_len"]
        self.traj_len = cfg["obs_len"] + self.pred_len
        self.emb_dim = cfg["emb_dim"]
        self.hid_dim = cfg["hid_dim"]
        self.drop = cfg["drop"]
        self.activ = cfg["activ"]
        self.scale = 1.0
        
        classifier_dims = [self.traj_len * self.inp_dim, self.hid_dim, int(self.hid_dim / 2), int(self.hid_dim / 4)]
        self.classifier = make_mlp(classifier_dims, activation = self.activ, dropout = self.drop)
        self.fc_out = nn.Linear(int(self.hid_dim/4), 1)
        
    def forward(self, traj, get_features = False, **kwargs):
        classifier_input = traj.contiguous().view(traj.size(0), -1)
        out = self.classifier(classifier_input)
        if get_features:
            return out
        out = self.fc_out(out)
        return out


class TemplVanDiscriminator(nn.Module):
    """ Temporal Vanilla network object
    """
    def __init__(self, cfg, inp_dim, device):
        super().__init__()
        self.device = device
        self.inp_dim = inp_dim
        self.traj_len = cfg["obs_len"] + cfg["pred_len"]
        self.emb_dim = cfg["emb_dim"]
        self.hid_dim = cfg["hid_dim"]
        self.drop = cfg["drop"]
        self.activ = cfg["activ"]
        self.net_type = cfg["net_type"]
        self.hidden_state = True if cfg["state"] == "hidden" else False
        self.scale = 1.0
        if self.activ == "prelu":
            self.activation =  nn.PReLU() 
        elif self.activ == "tanh":
            self.activation = nn.Tanh()
        elif self.activ == "relu":
            self.activation = nn.ReLU()
        elif self.activ == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError(self.activ)
        # encoder
        self.encoder = Encoder(self.inp_dim * self.traj_len, self.emb_dim, self.hid_dim, class_emb_dim = 0, net_type = self.net_type)
        
        classifier_dims = [self.hid_dim, int(self.hid_dim / 2), int(self.hid_dim / 4)]
        self.classifier = make_mlp(classifier_dims, activation = self.activ, dropout = self.drop)
        self.fc_out = nn.Linear(int(self.hid_dim/4), 1)
        
    def forward(self, traj, get_features = False, **kwargs):
        bs = traj.size(0)
        hidden_cell_state = ((torch.zeros(1, bs, self.hid_dim).to(self.device)),   
                             (torch.zeros(1, bs, self.hid_dim).to(self.device))) if self.net_type == "lstm" else torch.zeros(1, bs, self.hid_dim)
        traj = traj * self.scale
        out_emb = self.encoder.input_embedding(traj.contiguous().view(bs, -1))
        out_emb = out_emb.view(bs, -1, self.emb_dim)
        _, hidden_cell = self.encoder.temp_feat(out_emb, hidden_cell_state)
        hn = hidden_cell[0 if self.hidden_state else 1] if self.net_type == "lstm" else hidden_cell
        hn = hn.view(-1, self.hid_dim)
        if get_features:
            return self.activation(hn)
        out = self.classifier(hn)
        out = self.fc_out(out)
        return out


class FFSupCondDiscriminator(FFVanDiscriminator):
    """ Supervised conditioned discriminator based on a feed forward network
    """
    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)
        self.k_value = cfg["n_labels"]
        self.cond_type = cfg["condition_type"]
        self.emb_layer = LatentEmbedding(self.k_value, self.k_value) if self.cond_type == "embedding" else None
        if self.activ == "prelu":
            self.activation =  nn.PReLU() 
        elif self.activ == "tanh":
            self.activation = nn.Tanh()
        elif self.activ == "relu":
            self.activation = nn.ReLU()
        elif self.activ == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError(self.activ)
        
        
        self.classifier = nn.Sequential(OrderedDict([
            ("linear0", nn.Linear(self.traj_len * (self.inp_dim + self.k_value), self.emb_dim)),
            ("bn0", nn.BatchNorm1d(num_features = self.emb_dim)),
            ("activ0", self.activation),
            ("drop0", nn.Dropout(p = self.drop)),
            ("linear1", nn.Linear(self.emb_dim, self.hid_dim)), 
            ("bn1", nn.BatchNorm1d(num_features = self.hid_dim)),
            ("activ1", self.activation),
            ("drop1", nn.Dropout(p = self.drop)), 
            ("linear2", nn.Linear(self.hid_dim, int(self.hid_dim / 2))), 
            ("bn2", nn.BatchNorm1d(num_features = int(self.hid_dim / 2))),
            ("activ2", self.activation),
            ("drop2", nn.Dropout(p = self.drop)),
            ("linear3", nn.Linear(int(self.hid_dim / 2), int(self.hid_dim / 2))), 
            ("bn3", nn.BatchNorm1d(num_features = int(self.hid_dim / 2))),
            ("activ3", self.activation),
            ("drop3", nn.Dropout(p = self.drop))
        ]))
        self.fc_out = nn.Linear(int(self.hid_dim / 2), 1)
         
    def forward(self, traj, get_features = False, **kwargs):
        bs, size = traj.size(0), traj.size(1)
        labels = self.emb_layer(kwargs["labels"]) if self.cond_type == "embedding" else F.one_hot(kwargs["labels"], num_classes = self.k_value).float()
        labels = labels.unsqueeze(dim = 1).repeat(1, size, 1)
        
        inp = torch.cat([traj * self.scale, labels], dim = -1)
        classifier_input = inp.contiguous().view(bs, -1) 
        out = self.classifier.drop0(self.classifier.activ0(self.classifier.bn0(self.classifier.linear0(classifier_input))))
        out = self.classifier.drop1(self.classifier.activ1(self.classifier.bn1(self.classifier.linear1(out))))
        if get_features:
            return out
        out = self.classifier.drop2(self.classifier.activ2(self.classifier.bn2(self.classifier.linear2(out))))
        out = self.classifier.drop3(self.classifier.activ3(self.classifier.bn3(self.classifier.linear3(out))))
        out = self.fc_out(out)
        return out
    
    
class TempSupCondDiscriminator(TemplVanDiscriminator):
    """ Supervised conditioned discriminator based on temporal features
    """
    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)
        self.k_value = cfg["n_labels"]
        self.cond_type = cfg["condition_type"]
        self.emb_layer = LatentEmbedding(self.k_value, self.k_value) if self.cond_type == "embedding" else None
        self.encoder = Encoder(self.inp_dim * self.traj_len, self.emb_dim, self.hid_dim, class_emb_dim = self.k_value * self.traj_len, cat_start = True, net_type = self.net_type)
        self.fc_out = nn.Linear(int(self.hid_dim/4), 1)
    
    def forward(self, traj, get_features = False, **kwargs):
        bs, size = traj.size(0), traj.size(1)
        hidden_cell = ((torch.zeros(1, bs, self.hid_dim).to(self.device)),   
                       (torch.zeros(1, bs, self.hid_dim).to(self.device))) if self.net_type == "lstm" else torch.zeros(1, bs, self.hid_dim)
        
        labels = self.emb_layer(kwargs["labels"]) if self.cond_type == "embedding" else F.one_hot(kwargs["labels"], num_classes = self.k_value).float()
        labels = labels.unsqueeze(dim = 1).repeat(1, size, 1)
        
        # encoder spatial embedding
        inp_emb = torch.cat([traj * self.scale, labels], dim = -1)  
        out_emb = self.encoder.input_embedding(inp_emb.contiguous().view(bs, -1))
        out_emb = out_emb.view(bs, -1, self.emb_dim)
        
        # encoder temporal features
        _, hidden_cell = self.encoder.temp_feat(out_emb, hidden_cell)
        hn = hidden_cell[0 if self.hidden_state else 1] if self.net_type == "lstm" else hidden_cell
        hn = self.activation(hn.view(-1, self.hid_dim))
        if get_features:
            return hn
        
        # classifier
        out = self.classifier(hn)
        out = self.fc_out(out)
        return out
    
    
class FFSoftCondDiscriminator(FFSupCondDiscriminator):
    """ Soft conditioned discriminator based on a feed forward network
    """
    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)
        self.lay_ret_feat = cfg['layer_ret_feat']
        self.classifier = nn.Sequential(OrderedDict([
            ("linear0", nn.Linear(self.traj_len * self.inp_dim, self.emb_dim)),
            ("bn0", nn.BatchNorm1d(num_features = self.emb_dim)),
            ("activ0", self.activation),
            ("drop0", nn.Dropout(p = self.drop)),
            ("linear1", nn.Linear(self.emb_dim, self.hid_dim)), 
            ("bn1", nn.BatchNorm1d(num_features = self.hid_dim)),
            ("activ1",  self.activation),
            ("drop1", nn.Dropout(p = self.drop)), 
            ("linear2", nn.Linear(self.hid_dim, int(self.hid_dim / 2))), 
            ("bn2", nn.BatchNorm1d(num_features = int(self.hid_dim / 2))),
            ("activ2", self.activation),
            ("drop2", nn.Dropout(p = self.drop)),
            ("linear3", nn.Linear(int(self.hid_dim / 2), int(self.hid_dim / 2))), 
            ("bn3", nn.BatchNorm1d(num_features = int(self.hid_dim / 2))),
            ("activ3", self.activation),
            ("drop3", nn.Dropout(p = self.drop))
        ]))
        self.fc_out = LinearConditionalMaskLogits(int((self.hid_dim)/2), self.k_value)
        
    def forward(self, traj, get_features = False, **kwargs):
        bs = traj.size(0)
        traj = traj * self.scale
        classifier_input = traj.contiguous().view(bs, -1) 
        out = self.classifier.drop0(self.classifier.activ0(self.classifier.bn0(self.classifier.linear0(classifier_input))))
        if self.lay_ret_feat == "first" and get_features:
            return out
        out = self.classifier.drop1(self.classifier.activ1(self.classifier.bn1(self.classifier.linear1(out))))
        if self.lay_ret_feat == "second" and get_features:
            return out
        if self.lay_ret_feat == "third" and get_features:
            return out
        out = self.classifier.drop2(self.classifier.activ2(self.classifier.bn2(self.classifier.linear2(out))))
        if self.lay_ret_feat == "fourth" and get_features:
            return out
        out = self.classifier.drop3(self.classifier.activ3(self.classifier.bn3(self.classifier.linear3(out))))
        if self.lay_ret_feat == "fifth" and get_features:
            return out
        
        out = self.fc_out(out, kwargs["labels"])
        return out.unsqueeze(dim = -1)
    
    
class TempSoftCondDiscriminator(TemplVanDiscriminator):
    """ Soft conditioned discriminator based on tremporal features 
    """
    def __init__(self, cfg, inp_dim, device):
        super().__init__(cfg, inp_dim, device)
        self.k_value = cfg["n_labels"]
        classifier_dims = [self.hid_dim, int(self.hid_dim / 2), int(self.hid_dim / 4)]
        self.classifier = make_mlp(classifier_dims, activation = self.activ, dropout = self.drop)
        self.fc_out = LinearConditionalMaskLogits(int((self.hid_dim)/4), self.k_value)
        self.tanh = nn.Tanh()

    def forward(self, traj, get_features = False, **kwargs):
        bs = traj.size(0)
        hidden_cell = ((torch.zeros(1, bs, self.hid_dim).to(self.device)),   
                       (torch.zeros(1, bs, self.hid_dim).to(self.device))) if self.net_type == "lstm" else torch.zeros(1, bs, self.hid_dim)
        
        # encoder spatial embedding
        traj = traj * self.scale
        out_emb = self.encoder.input_embedding(traj.contiguous().view(bs, -1))
        out_emb = out_emb.view(bs, -1, self.emb_dim)
        
        # encoder temporal features
        _, hidden_cell = self.encoder.temp_feat(out_emb, hidden_cell)
        hn = hidden_cell[0 if self.hidden_state else 1] if self.net_type == "lstm" else hidden_cell
        hn = self.tanh(hn.view(-1, self.hid_dim))
        if get_features:
            return hn
        
        # classifier
        out = self.classifier(hn)
        
        out = self.fc_out(out,  kwargs["labels"])
        return out.unsqueeze(dim = -1)