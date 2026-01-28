"""
Embedding layer
    Act as a lookup table that maps token indices to dense vectors
"""
from .sliding_window_with_torch import Build_GPTDataLoader
import torch.nn as nn
import torch

# Sample input text, replace with the-verdict.txt
text = []

# Build DataLoader
context_length = 4
dataloader = Build_GPTDataLoader(
    txt= text, context_length= context_length, stride= context_length,
    batch_size= 8, shuffle= False,
)

# Gat a batch of data
data_iter = iter(dataloader)
inputs, targets = next(data_iter)


def build_embedding_layer(inputs, vocab_size= 50257, output_dim= 256, context_length= 4):
    """
    Build embedding layer for tokens and positions.

    Args:
        inputs (Tensor): Input tensor of token indices.
        vocab_size (int): Size of the vocabulary.
        output_dim (int): Dimension of the embedding vectors.
        context_length (int): Length of the input sequences.
    
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