import torch
import torch.nn as nn
from typing import List, OrderedDict


def get_noise(shape, noise_type, device):
    if noise_type == 'gaussian':
        return torch.randn(*shape, device=device)
    if noise_type == 'uniform':
        return torch.rand(*shape, device=device).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def make_mlp(dim_list : List[int], activation, batch_norm=False, dropout=0):
    """
    Generates MLP network:
    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_list : list, list containing activation function for each layer
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)
    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(negative_slope = 0.2))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'prelu':
            layers.append(nn.PReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class Encoder(nn.Sequential):
    """ Encoder object: input embedding + temporal features extractor    
    """
    def __init__(self, inp_dim : int, 
                       emb_dim : int, 
                       hid_dim : int, 
                       class_emb_dim : int,
                       cat_start = False, 
                       net_type  = "lstm"):
        emb_layers = [
            nn.Linear(inp_dim + class_emb_dim if cat_start else inp_dim, emb_dim),
            nn.PReLU(),
            ]
        input_embedding = nn.Sequential(*emb_layers)
        temp_feat = None
        if net_type:
            temp_feat = nn.LSTM(input_size = emb_dim + class_emb_dim if not cat_start else emb_dim, hidden_size = hid_dim, batch_first = True) if net_type == "lstm" \
                    else nn.GRU(input_size = emb_dim + class_emb_dim if not cat_start else emb_dim, hidden_size = hid_dim, batch_first = True)
        super().__init__(OrderedDict([
            ("input_embedding", input_embedding), 
            ("temp_feat", temp_feat)
        ]))
        
        
class Decoder(nn.Sequential):
    """ Decoder object: lstm + linear layer
    """
    def __init__(self, inp_dim :int,
                       emb_dim : int, 
                       hid_dim : int, 
                       out_dim : int, 
                       class_emb_dim : int, 
                       net_type = "lstm"):
        
        emb_layers = [
            nn.Linear(inp_dim + class_emb_dim, emb_dim),
            nn.PReLU(),
            ]
        input_embedding = nn.Sequential(*emb_layers)
        temp_feat = nn.LSTM(input_size = emb_dim, hidden_size = hid_dim, batch_first = True) if net_type == "lstm" \
                else nn.GRU(input_size = emb_dim, hidden_size = hid_dim, batch_first = True)
        predictor = nn.Linear(hid_dim, out_dim)
        super().__init__(OrderedDict([
            ("input_embedding", input_embedding),
            ("temp_feat", temp_feat),
            ("predictor", predictor)
        ]))
        
        
class LatentEmbedding(nn.Module):
    ''' projects class embedding onto hypersphere and returns the concat of the latent and the class embedding '''

    def __init__(self, nlabels, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(nlabels, embed_dim)

    def forward(self, y):
        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        return yembed


class LinearConditionalMaskLogits(nn.Module):
    ''' runs activated logits through fc and masks out the appropriate discriminator score according to class number
    '''
    def __init__(self, nc, nlabels):
        super().__init__()
        self.fc = nn.Linear(nc, nlabels)

    def forward(self, inp, y = None, take_best = False, get_features = False):
        out = self.fc(inp)
        if get_features: return out

        if not take_best:
            y = y.view(-1)
            index = torch.LongTensor(range(out.size(0)))
            if y.is_cuda:
                index = index.cuda()
            return out[index, y]
        else:
            # high activation means real, so take the highest activations
            best_logits, _ = out.max(dim=1)
            return best_logits
        