import torch.nn as nn
import torch

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, drop_rate, context_length, bias= False):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(drop_rate)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        _, num_tokens, _ = x.shape

        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        attn_scores = queries @ keys.transpose(-1, -2)

        # Apply causal mask to prevent attending to future tokens
        mask = self.mask.bool()[:num_tokens, : num_tokens]
        attn_scores_masked = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores_masked / (self.d_out ** 0.5),
            dim=-1
        )

        # Apply dropout to attention weights for regularization
        attn_weights_dropped = self.dropout(attn_weights)

        context_vec = attn_weights_dropped @ values
        return context_vec
