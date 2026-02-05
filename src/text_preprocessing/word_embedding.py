"""
Embedding layer
    Act as a lookup table that maps token indices to dense vectors
"""
import torch.nn as nn
import torch

def build_embedding_layer(inputs, vocab_size= 50257, output_dim= 256, context_length= 4):
    """
    Build embedding layer for tokens and positions.
 
    Returns:
        Tensor: Combined token and position embeddings.
    """
    token_embedding_layer = nn.Embedding(
        num_embeddings= vocab_size,
        embedding_dim= output_dim,
    )
    token_embeddings = token_embedding_layer(inputs)

    pos_embedding_layer = nn.Embedding(
        num_embeddings= context_length,
        embedding_dim= output_dim,
    )
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    input_embeddings = token_embeddings + pos_embeddings
    return input_embeddings