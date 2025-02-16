import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from tokenizer.load_BPE_tokenizer import load_bpe_tokenizer
from data_sampling.create_llm_dataset import LLMDataset
from data_sampling.get_llm_dataloader import LLMDataloader

class TestLLMDataLoader(unittest.TestCase):
    def setUp(self):
        self.file_path = 'data/sample_text.txt'
        self.tokenizer = load_bpe_tokenizer(encoding="o200k_base", model="gpt-4o").get_encoding_for_model(model="gpt-4o")
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        self.llm_dataset = LLMDataset(raw_text, self.tokenizer, max_length=4, stride=1)
        self.dataloader = LLMDataloader(self.llm_dataset, batch_size=1)

    def test_dataloader(self):
        for batch in self.dataloader:
            self.assertIsInstance(batch, list)
            self.assertGreater(len(batch), 0)

if __name__ == '__main__':
    unittest.main()