# Layer norm us used to improve stability and efficiency of neural networks
# It adjusts outputs of layer to have mean 0 and variance 1
# This speeds up convergence and ensures consistent and reliable training

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5

        # The scale and shift params are trainable params just for the model to decide how much to scale and shift the 
        # actual normalized values from data which will improve the training
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # uses unbiased estimator to calculate variance
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
