'''
A word has three projected representations in a sentence:
    Query: how this word seeks relevant information
    Key: how this word is represented for matching
    Value: the information this word provides if attended to
Therefore, it is beneficial to use different projections for Query, Key, and Value.
'''
import torch

# Sample input: 6 words with 3-dimensional embeddings
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)

# For reproducibility
torch.manual_seed(123)

"""
Establishing the query, key, value matrices 
W_q, W_k, W_v are projection matrices for whole input but not only for one word.
"""
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

# nn.Parameter makes it learnable
W_q = torch.nn.Parameter(torch.rand(d_in, d_out))  
W_k = torch.nn.Parameter(torch.rand(d_in, d_out))
W_v = torch.nn.Parameter(torch.rand(d_in, d_out))


# Computing the attention score for the second word
query_2 = x_2 @ W_q
keys = inputs @ W_k
values = inputs @ W_v

attn_scores_2 = query_2 @ keys.T

# Computing attention weights using softmax
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim= 0)