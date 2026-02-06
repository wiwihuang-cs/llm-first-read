import torch

def generate_text(model, idx, max_new_tokens, context_size):
    """
    Generate the number of max_new_tokens of text based on the input idx.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # During forward pass, PyTorch dynamically builds the computation graph 
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim= -1)
        idx_next = torch.argmax(probas, dim= -1, keepdim= True)
        idx = torch.cat((idx, idx_next), dim= 1)
    return idx