'''
Attention Score 

- The similarity between query and keys
- Commonly used methods: dot product(a method to compute the similarity between two vectors), ...

'''

import torch

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

query = inputs[1]  
attn_scores_2 = torch.empty(inputs.shape[0])
for i, key_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, key_i)

# print("Attention Scores:", attn_scores_2, sep= "\t")  # DEBUG: temporary output to verify the attention scores

'''
Normalization: a probability-like distribution
'''

# attn_scores_2_normalized = attn_scores_2 / attn_scores_2.sum()  # Softmax normalization have better numerical stability
attn_weights_2 = torch.softmax(attn_scores_2, dim= 0)

# print(attn_scores_2_normalized)  # DEBUG: tempoary output to verify the normalization

'''
Count the context vector: how a word is understood in the context of whole sentence 
'''

context_vec_2 = torch.zeros(inputs.shape[1])
for i, value in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]* value
