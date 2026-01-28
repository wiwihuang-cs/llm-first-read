import tiktoken
import torch
import torch.nn as nn

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

# Model configuration
cfg = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "drop_rate": 0.1,
    "num_layers": 12,
    "bias": False, 
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
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"], bias= cfg["bias"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"], bias= cfg["bias"]),
        )
        
    def forward(self, x):
        return self.layers(x)

# Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# Define GPT-2 model    
class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(cfg["num_layers"])]
        )

    def forward(self, x):
        batch_size, seq_length = x.shape

        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_length, device= x.device))
        x = tok_embeds + pos_embeds
        
        x = self.drop_emb(x)
        
        x = self.trf_blocks(x)  
        return x
