"""
Training and evaluation functions
"""

import torch


def evaluate_model(model, data_loader, criterion, device, stoi):
    """
    Evaluate model on a data loader and return average loss.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader for validation/test set
        criterion: Loss function
        device: Device to run on
        stoi: String to index mapping (for padding token)
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    vocab_size = len(stoi)
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            src_mask = (src == stoi['<PAD>'])
            tgt_mask = (tgt_input == stoi['<PAD>'])
            
            logits = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask
            )
            
            loss = criterion(logits.reshape(-1, vocab_size), tgt_target.reshape(-1))
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    model.train()
    return avg_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, device, stoi,
                num_epochs=150, print_every=50, eval_every=500):
    """
    Train the model
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        stoi: String to index mapping (for padding token)
        num_epochs: Number of training epochs
        print_every: Print loss every N batches
        eval_every: Evaluate on validation set every N batches
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    model.train()
    train_losses = []
    val_losses = []
    vocab_size = len(stoi)
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            src_mask = (src == stoi['<PAD>'])
            tgt_mask = (tgt_input == stoi['<PAD>'])
            
            optimizer.zero_grad()
            logits = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask
            )
            
            loss = criterion(logits.reshape(-1, vocab_size), tgt_target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % print_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_epoch_loss)
        
        avg_val_loss = evaluate_model(model, val_loader, criterion, device, stoi)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print("-"*60)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return train_losses, val_losses

