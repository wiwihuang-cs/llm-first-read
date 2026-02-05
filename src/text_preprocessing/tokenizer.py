"""
Tokenizer:
    Performs tokens segmentation and maps each token to a unique token id
    Requires because the embedding layer accepts only integer indices as input
"""

def encode_text_to_tokenIDs(texts, tokenizer):
    """
    Encode raw texts into tokenIDs.
    
    Returns:
        list of list of int
    """
    
    # The output of tokenizer.encode is a list of tokenIDs
    token_ids = [tokenizer.encode(text) for text in texts]
    return token_ids