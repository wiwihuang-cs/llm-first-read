"""
Tokenizer:
    Performs token segmentation and maps each token to a unique token idx
    Requires because the embedding layer accepts only integer indices as input
"""

def encode_text_to_tokenIDs(texts, tokenizer):
    """
    Encode raw texts into tokenID sequences.
    
    Returns:
        list of list of int: token ID sequences.
    """
    
    # The output of tokenizer.encode is a list of token IDs
    token_idx = [tokenizer.encode(txt) for txt in texts]
    return token_idx