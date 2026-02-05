import torch.nn as nn
import torch

# There're lots of benefits inheriting from nn.Module
class SelfAttention(nn.Module):
    # Self: the instance being created  
    def __init__(self, d_in, d_out):  
        super().__init__()
        """
        d_out is instance variable
        d_in is local variable in __init__() method
        """
        self.d_out= d_out  
        self.W_q = nn.Linear(d_in, d_out, bias=False)  
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

    # Builds the computation graph dynamically
    def forward(self, x):          
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        
        # The similarity between query and keys
        atte_scores = queries @ keys.T
        attn_weights = torch.softmax(
            atte_scores / (self.d_out ** 0.5), dim=-1
        )
        
        # Express the new representation of each token after attending the tokens in the sequence
        context_vec = attn_weights @ values
        return context_vec