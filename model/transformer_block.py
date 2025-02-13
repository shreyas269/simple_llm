import torch
import torch.nn as nn
from attention.multi_head_attention import MultiHeadAttention
from model.layer_norm import LayerNorm
from model.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg["emb_dim"], 
            d_out = cfg["emb_dim"], 
            context_length = cfg["context_length"], 
            dropout = cfg["dropout_rate"], 
            num_heads = cfg["num_heads"], 
            qkv_bias= cfg["qkv_bias"]
            )
        
        self.feedforward = FeedForward(cfg)

        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        
        x = x + shortcut
        shortcut = x

        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        
        x = x + shortcut
        return x
