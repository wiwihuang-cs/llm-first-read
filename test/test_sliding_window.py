import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from text_preprocessing.sliding_window import build_dataloader

def test_dataloader():
    texts = "Hello world! This is a test text for the sliding window dataloader."
    context_length = 3
    stride = 2
    batch_size = 2

    dataloader = build_dataloader(
        text= texts,
        context_length= context_length,
        stride= stride,
        batch_size= batch_size,
        shuffle= False,
        drop_last= False,
    )

    x, y = next(iter(dataloader))
    assert x.shape == (batch_size, context_length)
    assert y.shape == (batch_size, context_length)