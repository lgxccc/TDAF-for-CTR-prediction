
import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if 'pure_liner' in hparams:
        return MLP_Empty(input_shape[0], 128, hparams)
    elif len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    else:
        raise NotImplementedError

class MLP_Empty(nn.Module):
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP_Empty, self).__init__()
        self.n_outputs = n_inputs
        self.tmp = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x + self.tmp
        # return self.relu(x)
        return x


def Classifier(in_features, out_features, is_nonlinear=False, num_domains=None):
    if is_nonlinear:
        if in_features <= 4:
            return torch.nn.Linear(in_features, out_features)
        else:
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, in_features // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 2, in_features // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class ImprovedDomainAdaptor_Second(nn.Module):
    def __init__(self, hdim, out_dim, hparams, n_heads=16, MLP=None):
        super(ImprovedDomainAdaptor_Second, self).__init__()
        self.hdim = hdim
        self.out_dim = out_dim
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hdim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=hparams['attn_depth'])
        self.ln = nn.LayerNorm(hdim)
        self.mlp = MLP

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.ln(x)
        x = self.mlp(x)
        return x