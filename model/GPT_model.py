import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock
from model.layer_norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.transformer_blocks = nn.Sequential(* [TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.norm_layer = LayerNorm(cfg["emb_dim"])

        # This is the output layer that predicts the next token in the sequence using CE loss over logits generated over entire vocab
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, input):
        batch_size, seq_length = input.shape
        tok_embs = self.tok_embedding(input)
        pos_embs = self.pos_embedding(torch.arange(seq_length, device=input.device))
        x = tok_embs + pos_embs
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        
        # You probably do not need an attention_mask here (Check once) 
        # The attention mask is typically used within the TransformerBlock to prevent information leakage 
        # during the self-attention mechanism. 
        # The LayerNorm operation is applied independently to each position in the sequence
        #  and does not involve interactions between different positions.
        x = self.norm_layer(x) 
        logits = self.out_head(x)
        return logits
