import torch.nn as nn
import torch

class CausalAttention(nn.Module):
    def __init__(self, context_length, d_in, d_out, dropout, bias= False):
        super().__init__()
        self.d_out= d_out
        self.W_q= nn.Linear(d_in, d_out, bias=bias)
        self.W_k= nn.Linear(d_in, d_out, bias=bias)
        self.W_v= nn.Linear(d_in, d_out, bias=bias)
        self.dropout= nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length))
        )
    
    def forward(self, x):
        batch, num_tokens, d_in= x.shape

        keys= self.W_k(x)
        queries= self.W_q(x)
        values= self.W_v(x)

        attn_scores= queries @ keys.transpose(1, 2)
        attn_scores_masked= attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )

        attn_weights= torch.softmax(
            attn_scores_masked / (self.d_out ** 0.5),
            dim=-1
        )
        attn_weights_dropped= self.dropout(attn_weights)

        context_vec= attn_weights_dropped @ values
        return context_vec

inputs= torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)    
batch= torch.stack([inputs, inputs], dim= 0)
# print(batch.shape)  # DEBUG: temporary output to verify the batch shape

torch.manual_seed(123)
context_length= batch.shape[1]
ca = CausalAttention(d_in=batch.shape[2], d_out=3, context_length= context_length, dropout=0.0)
context_vecs = ca(batch)
print(context_vecs)  # DEBUG: temporary output to verify the context vectors