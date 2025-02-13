import torch
import torch.nn as nn

# Ideally in Llama and other GPT like models, as you increase the number of heads, it is recommended 
# to reduce the dimension of the context vectors by same scale

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # d_out here is the stacked dimension of the context vectors
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # This is the linear layer that combines the heads
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

        keys = keys.view(batches, num_tokens, self.num_heads, self.head_dim) # shape (batches, num_tokens, num_heads, head_dim)
        queries = queries.view(batches, num_tokens, self.num_heads, self.head_dim) # shape (batches, num_tokens, num_heads, head_dim)
        values = values.view(batches, num_tokens, self.num_heads, self.head_dim) # shape (batches, num_tokens, num_heads, head_dim)

        keys = keys.transpose(1, 2) # shape (batches, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2) # shape (batches, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2) # shape (batches, num_heads, num_tokens, head_dim)

        # Compute the attention scores for each head
        attn_scores = queries @ keys.transpose(2, 3) # shape (batches, num_heads, num_tokens, num_tokens)
        
        # Apply mask to the attention scores
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores/d_k**0.5, dim=-1) # shape (batches, num_heads, num_tokens, num_tokens)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values # shape (batches, num_heads, num_tokens, head_dim)
        context_vec = context_vec.transpose(1, 2) # shape (batches, num_tokens, num_heads, head_dim)

        # Combine the heads
        context_vec = context_vec.contiguous().view(batches, num_tokens, self.d_out) # shape (batches, num_tokens, d_out)
        # Comment: contiguous() is used to ensure that the tensor is stored in a contiguous chunk of memory
        context_vec = self.out_proj(context_vec) # Linear projection layer learns to combine information from different attention heads
        return context_vec
    