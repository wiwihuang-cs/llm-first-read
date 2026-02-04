import tiktoken
from text_preprocessing.tokenizer import encode_text_to_tokenIDs

# Sample input text
texts = [
    "just for test",
    "another test sentence",
]

# Build a tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

def test_tokenizer():
    token_idx= encode_text_to_tokenIDs(texts= texts, tokenizer= tokenizer)
    decoded_texts = [tokenizer.decode(token_id) for token_id in token_idx]
    assert decoded_texts == texts