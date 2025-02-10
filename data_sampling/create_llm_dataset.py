from torch.utils.data import Dataset

class LLMDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_text_chunk = token_ids[i:i + max_length]
            target_text_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(input_text_chunk)
            self.target_ids.append(target_text_chunk)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
