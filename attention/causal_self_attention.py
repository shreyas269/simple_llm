import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super.__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # register_buffer is used to store a tensor with model params that is not a learnable param
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
)

    def forward(self, x):
        batches, num_tokens, d_in = x.shape
        keys = self.W_key(x) # shape (batches, num_tokens, d_out)
        queries = self.W_query(x) # shape (batches, num_tokens, d_out)
        values = self.W_value(x) # shape (batches, num_tokens, d_out)
        attn_scores = queries @ keys.transpose(1, 2) # shape (batches, num_tokens, num_tokens)
        # Apply mask to the attention scores
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores/d_k**0.5, dim=-1) # shape (batches, num_tokens, num_tokens)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values # shape (batches, num_tokens, d_out)
        return context_vec