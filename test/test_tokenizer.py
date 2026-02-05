import tiktoken
from text_preprocessing.tokenizer import encode_text_to_tokenIDs

def test_tokenizer():
    # Sample input text
    texts = [
        "just for test",
        "another test sentence",
    ]

    # Build a tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids= encode_text_to_tokenIDs(texts= texts, tokenizer= tokenizer)
    decoded_texts = [tokenizer.decode(token_id) for token_id in token_ids]
    assert decoded_texts == texts