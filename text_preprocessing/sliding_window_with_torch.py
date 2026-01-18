import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):  # inherit Dataset so it can be used by PyTorch DataLoader
    def __init__(self, txt, tokenizer, max_length, stride):
        self.max_length = max_length
        self.stride = stride
        self.token_ids = tokenizer.encode(txt)

        # for i in range(0, len(token_ids)- max_length, stride):  # too eager

        #     input_chunk = token_ids[i: i + max_length]
        #     target_chunk = token_ids[i + 1: i + max_length + 1]
            
        #     self.input_ids.append(torch.tensor(
        #         input_chunk, 
        #         dtype = torch.long
        #     ))
        #     self.target_ids.append(torch.tensor(
        #         target_chunk, 
        #         dtype = torch.long
        #     ))
        
    def __len__(self):
        return (len(self.token_ids) - self.max_length) // self.stride
    
    def __getitem__(self, idx):
        i = idx* self.stride

        input_chunk = self.token_ids[i: i + self.max_length]
        target_chunk = self.token_ids[i + 1: i + self.max_length + 1]

        return (
            torch.tensor(input_chunk, dtype= torch.long),
            torch.tensor(target_chunk, dtype = torch.long),
        )
    
def Build_GPTDataLoader(txt, max_length= 256, stride= 128, 
                  batch_size= 4, shuffle= True, drop_last= True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
    )

    return dataloader

txt = "just for test"  
dataloader = Build_GPTDataLoader(txt, max_length=4, stride=1, batch_size=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
# print(first_batch)  # DEBUG: temporary output to verify the dataloader