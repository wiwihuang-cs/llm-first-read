from self_attention.self_attention import SelfAttention
import torch
import torch.nn as nn

def test_self_attention():
    d_in = 4
    d_out = 8
    self_attn = SelfAttention(
        d_in= d_in,
        d_out= d_out,
    )

    batch_size = 2
    token_num = 5
    x = nn.Parameter(torch.randn(batch_size, token_num, d_in))
    
    output = self_attn(x)
    assert output.shape == (batch_size, token_num, d_out)
    assert not torch.isnan(output).any()