'''
Attention Score 
    The similarity between query and keys
    Commonly used methods: dot product(a method to compute the similarity between two vectors), ...
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

# Computing the attention score for the second word
query = inputs[1]  
attn_scores_2 = torch.empty(inputs.shape[0])
for i, key_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, key_i)

# Normalization: a probability-like distribution
attn_weights_2 = torch.softmax(attn_scores_2, dim= 0)

# Counting the context vector: how a word is understood in the context of whole sentence 
context_vec_2 = torch.zeros(inputs.shape[1])
for i, value in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]* value
