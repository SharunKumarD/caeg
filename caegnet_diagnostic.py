import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import load_real_data, AcademicElectricityDataset
from model import CAEGNet

def train_and_track(ds_name, csv_path, epochs=15, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training CAEG-Net on {ds_name} Dataset ---")
    
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
    
    train_losses = []
    val_losses = []
    
    print(f"Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for step, (x_load, x_cal, y_load, _) in enumerate(train_loader):
            x_load, y_load = x_load.to(device), y_load.to(device)
            optimizer.zero_grad()
            
            if step == 0:
                batch_prev_error = torch.zeros(batch_size, 24).to(device)
            
            predictions, _ = model(x_load, batch_prev_error)
            loss = criterion(predictions, y_load)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                batch_prev_error = (predictions - y_load).detach()
                
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
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
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

def main():
    epochs = 15
    datasets = [
        ("ELEC2", "data/dataset_elec2.csv"),
        ("Modern PJM", "data/dataset_modern.csv")
    ]
    
    all_results = {}
    for ds_name, csv_path in datasets:
        t_losses, v_losses = train_and_track(ds_name, csv_path, epochs=epochs)
        all_results[ds_name] = (t_losses, v_losses)
        
    print("\nGenerating side-by-side plot...")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, ds_name in enumerate(["ELEC2", "Modern PJM"]):
        t_losses, v_losses = all_results[ds_name]
        axs[i].plot(range(1, epochs + 1), t_losses, label='Training Loss', marker='o', color='blue')
        axs[i].plot(range(1, epochs + 1), v_losses, label='Validation Loss', marker='o', color='orange')
        axs[i].set_title(f'CAEG-Net Convergence ({ds_name})')
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Loss (MSE)')
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].legend()
        
    plt.tight_layout()
    out_path = 'caegnet_convergence.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot successfully saved to {out_path}")

if __name__ == "__main__":
    main()
