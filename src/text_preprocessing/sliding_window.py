import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Inherit Dataset so it can be used by PyTorch DataLoader
class GPTDataset(Dataset):  
    def __init__(self, text, tokenizer, context_length, stride):
        self.token_idx = tokenizer.encode(text)
        self.context_length = context_length
        self.stride = stride
    
    # ============================
    # Two essential methods for Dataset
    # ============================    
    def __len__(self):
        return (len(self.token_idx) - self.context_length) // self.stride
    
    def __getitem__(self, idx):
        i = idx * self.stride

        input_chunk = self.token_idx[i: i + self.context_length]
        target_chunk = self.token_idx[i + 1: i + self.context_length + 1]

        return (
            torch.tensor(input_chunk, dtype= torch.long),
            torch.tensor(target_chunk, dtype = torch.long),
        )
    
def build_dataloader(text, context_length= 256, stride= 128, 
                  batch_size= 4, shuffle= True, drop_last= True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, context_length, stride)


    """
    DataLoader collects batch_size samples by calling Dataset.__getitem__()
    and stacks them into batched tensors.
    """
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,  
        shuffle = shuffle,
        drop_last = drop_last,
    )
    return dataloader
