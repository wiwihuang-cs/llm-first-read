from self_attention.causal_attention import CausalAttention
import torch
import torch.nn as nn

def test_causal_attention():
    d_in = 4
    d_out = 8
    drop_rate = 0.1
    context_length = 10
    causal_attn = CausalAttention(
        d_in= d_in,
        d_out= d_out,
        drop_rate= drop_rate,
        context_length= context_length,
    )

    batch_size = 2
    token_num = 5
    x = nn.Parameter(torch.randn(batch_size, token_num, d_in))

    output = causal_attn(x)
    assert output.shape == (batch_size, token_num, d_out)
    assert not torch.isnan(output).any()
    # assert False, f"{output}"