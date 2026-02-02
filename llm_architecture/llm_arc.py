import tiktoken
import torch
import torch.nn as nn
from self_attention.multi_attention import MultiHeadAttention

#%%
# Sample input texts
texts = [
    "Every effort moves you",
    "Every day holds a",
]

# Initialize GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

def preprocess_text(texts, tokenizer):
    """
    Convert raw texts into token ID sequences and stack them into a batch tensor.
    
    Args: 
        texts (list of str): raw input texts.
    
    Returns:
        torch.Tensor: batch tensor of token ID sequences.
    """

    # Tokenize raw text into token ID sequences
    token_ids = [torch.tensor(tokenizer.encode(txt)) for txt in texts]

    # Stack token ID sequences into a batch tensor
    batch = torch.stack(token_ids, dim= 0) 
    
    # DEBUG: temporary output to verify tokenization
    # print(batch)  
    return batch

#%%
# Model configuration
cfg = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "drop_rate": 0.1,
    "num_layers": 12,
    "bias": False, 
    "heads": 12,
}

# ============================
# Define feed-forward network
# ============================
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (MLP):
    Consists of two linear layers with a non-linear activation in between.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # Apply linear projection on the last (embedding) dimension
            # (batch_size, seq_length, emb_dim) -> (batch_size, seq_length, 4* emb_dim)
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"], bias= cfg["bias"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"], bias= cfg["bias"]),
        )
        
    def forward(self, x):
        return self.layers(x)
    
# Define the layer normalization
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eps= 1e-5
        self.scale= nn.Parameter(torch.ones(cfg["emb_dim"]))
        self.shift= nn.Parameter(torch.zeros(cfg["emb_dim"]))

    def forward(self, x):
        # The keepdim= True keeps the dimensions be reduced
        # And the shape of mean will be (batch_size, seq_length, 1)
        # So that the broadcasting can be applied correctly, 
        # (batch_size, seq_length, emb_dim) - (batch_size, seq_length, 1) -> (batch_size, seq_length, emb_dim)
        mean= x.mean(-1, keepdim= True)
        var= x.var(-1, keepdim= True, unbiased= False)
        norm_x= (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
# Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in= cfg["emb_dim"],
            d_out= cfg["emb_dim"],
            heads= cfg["heads"],
            context_length= cfg["context_length"],
            drop= cfg["drop_rate"],
            bias= cfg["bias"],
        )
        self.ffn = FeedForward(cfg= cfg)
        self.norm1 = LayerNorm(cfg= cfg)
        self.norm2 = LayerNorm(cfg= cfg)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        short_cut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + short_cut

        short_cut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + short_cut
        return x

class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg= cfg) for _ in range(cfg["layers"])]
        )

        self.final_norm = LayerNorm(cfg= cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias= cfg["bias"])

    def forward(self, x):
        batch_size, seq_length = x.shape

        # nn.Embedding preserves the input shape and appends the embedding dimension
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_length, device= x.device))

        # (batch_size, seq_length, emb_dim) + (seq_length, emb_dim) -> (batch_size, seq_length, emb_dim) via broadcasting
        x = tok_embeds + pos_embeds
        
        # Randomly zero out some elements, which chosen from (batch_size* seq_length* emb_dim), for regularization
        x = self.drop_emb(x)
        
        x = self.trf_blocks(x)

        x = self.final_norm(x)
        x = self.out_head(x)

        return x