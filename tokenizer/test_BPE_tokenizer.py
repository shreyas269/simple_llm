import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from tokenizer.load_BPE_tokenizer import load_bpe_tokenizer

class TestBPETokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = load_bpe_tokenizer(encoding="o200k_base", model="gpt-4o")
        self.encoding = self.tokenizer.get_encoding(encoding="o200k_base")
        self.encoding_for_model = self.tokenizer.get_encoding_for_model(model="gpt-4o")

    def test_encoding(self):
        text = "Hello, world!"
        encoded = self.encoding.encode(text)
        expected_encoded = [13225, 11, 2375, 0]
        self.assertEqual(encoded, expected_encoded)
    
    def test_encoding_for_model(self):
        text = "Hello, world!"
        encoded = self.encoding_for_model.encode(text)
        expected_encoded = [13225, 11, 2375, 0]
        self.assertEqual(encoded, expected_encoded)

    def test_decoding(self):
        tokens = [13225, 11, 2375, 0]
        decoded = self.encoding.decode(tokens)
        expected_decoded = "Hello, world!"
        self.assertEqual(decoded, expected_decoded)

    def test_decoding_for_model(self):
        tokens = [13225, 11, 2375, 0]
        decoded = self.encoding_for_model.decode(tokens)
        expected_decoded = "Hello, world!"
        self.assertEqual(decoded, expected_decoded)

if __name__ == '__main__':
    unittest.main()