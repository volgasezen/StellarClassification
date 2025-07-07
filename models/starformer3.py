# %%
import sys
sys.path.insert(1, r'/home/oban/Desktop/Volga/stellar-classification/ecgdetr/models')

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from backbone import build_backbone
# from rotary_embedding_torch import RotaryEmbedding

# %%
class tAPE(nn.Module): # Equation 13 page 11
    def __init__(self, d_model, dropout=0.1, max_len=835, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class Conv1dEmbedder(nn.Module):
    def __init__(self, seq_len=64, patch_size=4, hidden_d=64):
        super(Conv1dEmbedder, self).__init__()
        self.backbone = build_backbone(1,hidden_d)
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.hidden_d = hidden_d
        
        assert seq_len % patch_size == 0, "Input shape not entirely divisible by patch size"
        self.n_patches = seq_len // patch_size
    
        self.tape = tAPE(hidden_d, max_len=2369)
    
        self.linear_mapper = nn.Linear(self.patch_size, self.hidden_d)
    
        self.temp_token = nn.Parameter(torch.rand(1, self.hidden_d))
        self.lum_token = nn.Parameter(torch.rand(1, self.hidden_d))

    def forward(self, fluxes):
        batch = fluxes.shape[0]
        features = self.backbone(fluxes)
        patches = features.reshape(batch,-1,self.patch_size)
        tokens = self.linear_mapper(patches)
        tokens = torch.stack([torch.cat((self.temp_token, self.lum_token, tokens[i]),dim=0) for i in range(len(tokens))])
        tokens = self.tape(tokens)
        return tokens

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout, d_out):
        super(Transformer, self).__init__()

        self.embed = Conv1dEmbedder(hidden_d=d_model)
        
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward = 2048, 
            activation = 'relu', 
            layer_norm_eps = 1e-5,
            batch_first=True
        )

        self.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer = self.TransformerEncoderLayer,
            num_layers = num_layers
        )

        self.out_layer = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1) #insert channel dim
        x = self.embed(x)
        x = self.TransformerEncoder(x)
        if x.shape[0] != 1:
            x = x.squeeze()
        x = x[:, 0] #look at cls token only
        x = F.relu(self.dropout(x))
        x = self.out_layer(x)
        return x
# %%
