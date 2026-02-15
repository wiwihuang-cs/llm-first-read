import torch
import torch.nn as nn

def calc_loss_batch(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # Forward pass
    logits = model(input_batch)

    """
    Cross entropy is used for multi-class classification problems, 
    and it expects the input to be of shape (batch_size, num_classes) and the target to be of shape (batch_size).
    """
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(model, batches_num, dataloader, device):
    total_loss = 0.0
    if batches_num is None:
        batches_num = len(dataloader)
    else:
        batches_num = min(batches_num, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < batches_num:
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / batches_num

def evaluate_model(model, train_dataloader, val_dataloader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(model, eval_iter, train_dataloader, device)
        val_loss = calc_loss_loader(model, eval_iter, val_dataloader, device)
    model.train()
    return train_loss, val_loss


def train_model(model, train_dataloader, val_dataloader, epochs_num, device, optimizer,
                eval_freq, eval_iter):
    train_losses, val_losses = [], []
    global_step = 0
    for epoch in range(epochs_num):
        model.train()
        for input_batch, target_batch in train_dataloader:

            # Forward pass and calculate loss 
            loss = calc_loss_batch(model, input_batch, target_batch, device)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update parameters per batch
            optimizer.step()
            global_step += 1
            
            # Validate the model
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_dataloader, val_dataloader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{epochs_num}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses
