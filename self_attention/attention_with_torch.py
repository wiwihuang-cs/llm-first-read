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

        """
        In Deep Learning, linear transformation is represented as y = Wx + b
        In PyTorch, nn.Linear is a module that compute y = x*W^T + b ((Wx)^T = x^T*W^T)
        So, It create A with shape (out_features, in_features)
        """
        self.W_q = nn.Linear(d_in, d_out, bias=False)  
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

    """
    The only method needed to be defined
    Builds the computation graph dynamically
    """
    def forward(self, x):          
        """
        Originally, only the method can be called,
        But, python introduces __call__() so that we can call the instance directly  
        """
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        
        atte_scores = queries @ keys.T
        attn_weights = torch.softmax(
            atte_scores / (self.d_out ** 0.5), dim=-1
        )
        
        context_vec = attn_weights @ values
        return context_vec