import unittest
from tokenizer.simple_tokenizer import SimpleTokenizer

class TestSimpleTokenizer(unittest.TestCase):
    def setUp(self):
        # Sample vocabulary
        self.vocab = {"hello": 0, "world": 1, "!": 2, ",": 3}
        self.tokenizer = SimpleTokenizer(self.vocab)
        self.tokenizer.add_special_tokens()
    
    def test_encode_known_tokens(self):
        text = "hello, world!"
        expected_output = [0, 3, 1, 2]
        self.assertEqual(self.tokenizer.encode(text), expected_output)
    
    def test_encode_unknown_tokens(self):
        text = "hello universe!"
        expected_output = [0, self.tokenizer.str_to_int["<|unk|>"], 2]
        self.assertEqual(self.tokenizer.encode(text), expected_output)
    
    def test_decode(self):
        token_ids = [0, 3, 1, 2]
        expected_output = "hello, world!"
        self.assertEqual(self.tokenizer.decode(token_ids), expected_output)
    
    def test_decode_with_unknown_token(self):
        token_ids = [0, self.tokenizer.str_to_int["<|unk|>"], 2]
        expected_output = "hello <|unk|>!"
        self.assertEqual(self.tokenizer.decode(token_ids), expected_output)
    
    def test_special_tokens(self):
        self.assertIn("<|endoftext|>", self.tokenizer.str_to_int)
        self.assertIn("<|unk|>", self.tokenizer.str_to_int)

if __name__ == "__main__":
    unittest.main()