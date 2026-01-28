"""
Sliding window
    Used to generate input-target pair
"""
import tiktoken

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
    return token_ids

def sliding_window(token_ids, context_length= 4):
    """
    Generate input-target pair using sliding window.
    
    Args:
        token_ids: list of int
        context_length: int
    
    Returns:
        input: list of int
        target: list of int
    """
    
    input = token_ids[:context_length]
    target = token_ids[1:context_length+1]

    return input, target