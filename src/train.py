"""
CAEG-Net Training Engine
Handles PyTorch model training with closed-loop error feedback state tracking,
learning rate scheduling, early stopping, and loss logging.
"""

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    device: str = "cpu",
    model_name: str = "caeg_net",
    save_dir: str = "checkpoints"
) -> Dict[str, Any]:
    """
    Trains PyTorch forecasting model with closed-loop feedback tracking.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "epoch_times": []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        # Track online exponential moving average (EMA) of error for closed loop
        recent_error_state = torch.zeros((train_loader.batch_size, 1), device=device)
        
        for batch in train_loader:
            x_seq = batch["x_seq"].to(device)
            y_seq = batch["y_seq"].to(device)
            raw_x_load = batch["raw_x_load"].to(device)
            
            # Ensure correct batch size for recent_error tensor
            if recent_error_state.size(0) != x_seq.size(0):
                recent_error_state = torch.zeros((x_seq.size(0), 1), device=device)
                
            optimizer.zero_grad()
            
            # Forward pass
            fused_pred, expert_weights, _ = model(
                x_seq=x_seq, 
                raw_x_load=raw_x_load, 
                recent_error=recent_error_state.detach()
            )
            
            loss_mse = criterion(fused_pred, y_seq)
            loss_mae = mae_criterion(fused_pred, y_seq)
            
            # Entropy balance penalty to prevent early expert collapse
            avg_weights = torch.mean(expert_weights, dim=0)
            entropy_penalty = -torch.sum(avg_weights * torch.log(avg_weights + 1e-8))
            
            # Combined Loss
            total_loss = loss_mse + 0.05 * loss_mae - 0.005 * entropy_penalty
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss_mse.item()
            train_batches += 1
            
            # Update closed-loop recent error state (EMA update)
            with torch.no_grad():
                batch_mae = torch.mean(torch.abs(fused_pred - y_seq), dim=1, keepdim=True)
                recent_error_state = 0.8 * recent_error_state + 0.2 * batch_mae
                
        avg_train_loss = train_loss_sum / train_batches
        
        # Validation Loop
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_batches = 0
        val_recent_error = torch.zeros((val_loader.batch_size, 1), device=device)
        
        with torch.no_grad():
            for batch in val_loader:
                x_seq = batch["x_seq"].to(device)
                y_seq = batch["y_seq"].to(device)
                raw_x_load = batch["raw_x_load"].to(device)
                
                if val_recent_error.size(0) != x_seq.size(0):
                    val_recent_error = torch.zeros((x_seq.size(0), 1), device=device)
                    
                fused_pred, _, _ = model(
                    x_seq=x_seq, 
                    raw_x_load=raw_x_load, 
                    recent_error=val_recent_error
                )
                
                v_mse = criterion(fused_pred, y_seq).item()
                v_mae = mae_criterion(fused_pred, y_seq).item()
                
                val_loss_sum += v_mse
                val_mae_sum += v_mae
                val_batches += 1
                
                # Update val error state
                batch_mae = torch.mean(torch.abs(fused_pred - y_seq), dim=1, keepdim=True)
                val_recent_error = 0.8 * val_recent_error + 0.2 * batch_mae

        avg_val_loss = val_loss_sum / val_batches
        avg_val_mae = val_mae_sum / val_batches
        epoch_dur = time.time() - epoch_start
        
        scheduler.step(avg_val_loss)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_mae"].append(avg_val_mae)
        history["epoch_times"].append(epoch_dur)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save(model.state_dict(), ckpt_path)

    total_training_time = time.time() - start_time
    history["total_training_time"] = total_training_time
    history["best_val_loss"] = best_val_loss
    
    return history
