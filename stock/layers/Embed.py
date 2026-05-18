import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)   
        self.linear = nn.Linear(nf, target_window) 
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        x = self.flatten(x)          
        x = self.linear(x)            
        x = self.dropout(x)
        return x

class PatchEmbedding_group(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout, feature_group):
        super(PatchEmbedding_group, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.feature_group = feature_group
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        self.fc = nn.Linear(patch_len, d_model)

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.ModuleList()
        for i in range(len(self.feature_group)):
            layer = nn.Linear(d_model * len(self.feature_group[i]), d_model, bias=False)
            self.value_embedding.append(layer)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.fc(x) # patch len -> d_model
        x = x.permute(0, 3, 2, 1)
        out = []
        for index in range(len(self.feature_group)):
            out.append(x[:,:,:,self.feature_group[index]])
            out[index] = torch.reshape(out[index], (out[index].shape[0], out[index].shape[2], out[index].shape[3] * out[index].shape[1]))
            out[index] = self.value_embedding[index](out[index]) + self.position_embedding(out[index])
            out[index] = self.dropout(out[index])
            # x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[3] * x[:,:,:,self.feature_group[index]]))
        # x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[3] * x.shape[1])) # patch_len * n_vars
        # Input encoding
        # x = self.value_embedding(x) + self.position_embedding(x)
        return out, n_vars
    
