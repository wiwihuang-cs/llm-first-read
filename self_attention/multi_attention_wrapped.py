import torch.nn as nn 
from causal_attention import CausalAttention
import torch

class MultiHeadAttention_wrapped(nn.Module):
    def __init__(self, d_in, d_out, dropout, context_length, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in= d_in, d_out= d_out, dropout=dropout, context_length=context_length) 
             for _ in range(num_heads)]
        )
    
    def forward(self, x):
        # Concatenate the outputs from all attention heads
        return torch.cat([head(x) for head in self.heads], dim= -1)