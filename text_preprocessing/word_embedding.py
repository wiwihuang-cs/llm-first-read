"""
Embedding layer

- Mapping each token ID to the a uniquo vector 
"""
from .sliding_window_with_torch import Build_GPTDataLoader
import torch.nn as nn

txt = "just for test"
max_length = 4
dataloader = Build_GPTDataLoader(
    txt= txt,max_length= max_length, stride= max_length,
    batch_size= 8, shuffle= False,
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# print(inputs)  # temporary output to verify the dataloader
# print(inputs.shape)

vocab_size = 50257  # GPT-2 vocabulary size
output_dim = 256
token_embedding_layer = nn.Embedding(
    num_embeddings= vocab_size,
    embedding_dim= output_dim,
)
token_embeddings = token_embedding_layer(inputs)

# print(token_embeddings.shape)  # temporary output to verify the embedding layer