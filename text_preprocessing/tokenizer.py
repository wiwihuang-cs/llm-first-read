"""
Tokenizer:
    Performs token segmentation and maps each token to a unique token ID
    Requires because the embedding layer accepts only integer indices as input
"""

# BPE algorithm by GPT models
import tiktoken  

# Sample input text
text = "just for test"

# Build a tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

def encode_text_to_tokenIDs(texts, tokenizer):
    """
    Encode raw texts into tokenID sequences.

    Args:
        texts (list of str): raw input texts.
        tokenizer: tokenizer instance.
    
    Returns:
        list of list of int: token ID sequences.
    """
    
    token_ids = [tokenizer.encode(txt) for txt in texts]

    # DEBUG: temporary output to verify the tokenizer encoding
    # print(token_ids) 

    raw_texts = [tokenizer.decode(token_id) for token_id in token_ids]
    
    # DEBUG: tempoary output to verify the tokenizer decoding
    # print(raw_texts) 

    return token_ids