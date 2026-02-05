import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, heads, context_length, drop= 0.1, bias= False):
        super().__init__()
        assert d_out % heads == 0, "d_out must be divisible by number of heads"

        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        
        self.dropout = nn.Dropout(drop)
        self.register_buffer(
            'mask',

            # the diagnal=1 ensures that the current token can also attend to itself
            torch.tril(torch.ones(context_length, context_length), diagonal= 1)            
        )

        # Multi-head settings
        self.heads = heads
        self.d_heads = d_out // heads

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # Split the last dimension into heads
        keys = keys.view(batch_size, num_tokens, self.heads, self.d_heads)
        queries = queries.view(batch_size, num_tokens, self.heads, self.d_heads)
        values = values.view(batch_size, num_tokens, self.heads, self.d_heads)

        # Move heads forward for attention: (batch_size, heads, num_tokens, d_heads)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        causal_attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(causal_attn_scores / (self.d_heads ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Calculate context vectors
        context_vec = (attn_weights @ values)

        # (batch_size, num_tokens, heads, d_heads)
        context_vec = context_vec.transpose(1, 2) 

        # Merge the heads 
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out) 
        return context_vec