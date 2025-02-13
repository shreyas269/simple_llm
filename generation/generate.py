import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizer.load_BPE_tokenizer import load_bpe_tokenizer
from model.GPT_model import GPTModel

def generate_text(model, batched_input, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # Truncate the context here if necessary
        batched_input_trunc = batched_input[:, -context_size:]

        with torch.no_grad():
            logits = model(batched_input_trunc) # model output shape: (batch_size, seq_length, vocab_size)

        # Take the logits at the last position and get the next token by sampling from the distribution
        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)
        pred_next_token = torch.argmax(probs, dim=-1, keepdim=True)
        batched_input = torch.cat((batched_input, pred_next_token), dim=1)

    return batched_input
    
def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768, # Embedding dimension
        "num_heads": 12, # Number of attention heads
        "num_layers": 12, # Number of layers
        "dropout_rate": 0.1, # Dropout rate
        "qkv_bias": False # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    start_context = "Hello, I am"
    tokenizer = load_bpe_tokenizer(encoding="o200k_base", model="gpt-4o").get_encoding_for_model(model="gpt-4o")
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()
    out = generate_text(
        model=model,
        batched_input=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

if __name__ == "__main__":
    main()