import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset import load_real_data, AcademicElectricityDataset
from model import CAEGNet

def train_on_dataset(ds_name, csv_path, save_path, epochs, batch_size, patience, device):
    print(f"\n{'='*50}")
    print(f"   Training CAEG-Net on {ds_name} Dataset")
    print(f"{'='*50}")
    
    X_load, X_cal, Y_load, scaler = load_real_data(csv_path=csv_path, seq_len=168, pred_len=24, train_ratio=0.8)
    
    total_samples = len(X_load)
    train_size = int(0.8 * total_samples)
    
    train_dataset = AcademicElectricityDataset(X_load[:train_size], X_cal[:train_size], Y_load[:train_size])
    val_dataset = AcademicElectricityDataset(X_load[train_size:], X_cal[train_size:], Y_load[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    model = CAEGNet(input_dim=1, seq_len=168, pred_len=24).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"Starting Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples...")
    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        train_loss = 0
        
        for step, (x_load, x_cal, y_load, _) in enumerate(train_loader):
            x_load, y_load = x_load.to(device), y_load.to(device)
            
            optimizer.zero_grad()
            
            # During training, we don't have true sequential prev_error because of batching,
            # but since we set shuffle=False, we can persist it across steps.
            # For simplicity, we zero it on the first step of epoch.
            if step == 0:
                batch_prev_error = torch.zeros(batch_size, 24).to(device)
            
            predictions, gating_weights = model(x_load, batch_prev_error)
            loss = criterion(predictions, y_load)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                batch_prev_error = (predictions - y_load).detach()
                
        train_loss /= len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_prev_error = torch.zeros(batch_size, 24).to(device)
            for x_load, x_cal, y_load, _ in val_loader:
                x_load, y_load = x_load.to(device), y_load.to(device)
                curr_batch_size = x_load.size(0)
                
                if curr_batch_size != val_prev_error.size(0):
                    val_prev_error = val_prev_error[:curr_batch_size]
                
                predictions, _ = model(x_load, val_prev_error)
                loss = criterion(predictions, y_load)
                val_loss += loss.item()
                val_prev_error = (predictions - y_load).detach()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Validation loss improved. Best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"  [!] No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"==> Early stopping triggered after {epoch+1} epochs.")
            break

def train_caeg_net(epochs=15, batch_size=32, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    datasets = [
        ("ELEC2", "data/dataset_elec2.csv", "caeg_net_elec2_best.pth"),
        ("Modern PJM", "data/dataset_modern.csv", "caeg_net_modern_best.pth")
    ]
    
    for ds_name, csv_path, save_path in datasets:
        train_on_dataset(ds_name, csv_path, save_path, epochs, batch_size, patience, device)

if __name__ == "__main__":
    train_caeg_net()
