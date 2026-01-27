import tiktoken
import torch
import torch.nn as nn

# Initilize GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Tokenize raw text into token ID sequences
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch = []
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

# Stack token ID sequences into a batch tensor
batch = torch.stack(batch, dim= 0)
# print(batch)  # DEBUG: temporary output to verify tokenization

cfg = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "drop_rate": 0.1,
    "num_layers": 12,
    "bias": False, 
}

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Position-wise feed-forward network (MLP):
        # two linear layers with a non-linear activation in between.
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"], bias= cfg["bias"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"], bias= cfg["bias"]),
        )
        
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Act like a lookup table that maps token ID sequences to dense vectors
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

model = GPT2Model(cfg)
# print(model(batch))  # DEBUG: temporary output to verify model output