import re

class SimpleTokenizer:
    def __init__(self, vocab): # vocab is a dictionary that maps words to integers. vocab ~ {s:i}
        self.str_to_int = vocab # Stored as class attribute
        self.int_to_str = {i:s for s,i in vocab.items()}

    def add_special_tokens(self):
        # Add special tokens to the vocabulary
        self.str_to_int["<|endoftext|>"] = len(self.str_to_int)
        self.str_to_int["<|unk|>"] = len(self.str_to_int)
        # Modify the int_to_str dictionary as well
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}

    def encode(self, text):
        # Split the text into a list of strings
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Now remove whitespaces
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Replace unknown tokens with <|unk|>
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        # Join the tokens by whitespace
        text = " ".join([self.int_to_str[id] for id in ids])
        # Removes spaces before the specified punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text