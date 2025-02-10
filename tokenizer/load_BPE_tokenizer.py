import tiktoken

class load_bpe_tokenizer:
    def __init__(self, encoding="o200k_base", model="gpt-4o"):
        self.encoding = encoding
        self.model = model

    def get_encoding(self, encoding):
        if encoding:
            enc = tiktoken.get_encoding(encoding)
        else:
            enc = tiktoken.get_encoding(self.encoding)
        return enc
    
    def get_encoding_for_model(self, model):
        if model:
            enc = tiktoken.encoding_for_model(model)
        else:
            enc = tiktoken.encoding_for_model(self.model)
        return enc
    
