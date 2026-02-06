from self_attention.multi_attention import MultiHeadAttention
import torch

def test_multi_head_attention():
    d_in = 2
    d_out = 16
    heads = 4
    context_length = 5
    drop_rate = 0.1

    multi_attn = MultiHeadAttention(
        d_in= d_in,
        d_out= d_out,
        heads= heads,
        context_length= context_length,
        drop_rate= drop_rate
    )

    batch_size = 2
    num_tokens = 5
    x = torch.randn(batch_size, num_tokens, d_in)

    output = multi_attn(x)
    assert output.shape == (batch_size, num_tokens, d_out)
    assert not torch.isnan(output).any()
    # assert False, f"{output}"