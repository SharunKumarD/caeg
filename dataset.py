import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def load_real_data(csv_path='data/dataset.csv', seq_len=168, pred_len=24, train_ratio=0.8):
    print(f"Loading real academic dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    load_values = df['load'].values.reshape(-1, 1)
    
    # Fit StandardScaler EXCLUSIVELY on training split to prevent data leakage
    train_size = int(len(load_values) * train_ratio)
    scaler = StandardScaler()
    scaler.fit(load_values[:train_size])
    
    scaled_load = scaler.transform(load_values)
    
    # Extract explicit calendar features
    # 1. Normalized hour of day (0 to 1)
    hour_val = df['datetime'].dt.hour.values + df['datetime'].dt.minute.values / 60.0
    norm_hour = hour_val / 24.0
    
    # 2. Normalized day of week (0 to 1)
    norm_day = df['datetime'].dt.dayofweek.values / 6.0
    
    calendar_features = np.stack([norm_hour, norm_day], axis=-1)
    
    X_load, X_cal, Y_load = [], [], []
    
    # Generate PyTorch tensors via sliding window
    total_len = len(scaled_load) - seq_len - pred_len
    for i in range(total_len):
        X_load.append(scaled_load[i : i + seq_len])
        X_cal.append(calendar_features[i : i + seq_len])
        Y_load.append(scaled_load[i + seq_len : i + seq_len + pred_len, 0])
        
    return (torch.tensor(np.array(X_load), dtype=torch.float32),
            torch.tensor(np.array(X_cal), dtype=torch.float32),
            torch.tensor(np.array(Y_load), dtype=torch.float32),
            scaler)

class AcademicElectricityDataset(Dataset):
    def __init__(self, X_load, X_cal, Y_load):
        self.X_load = X_load
        self.X_cal = X_cal
        self.Y_load = Y_load
        
    def __len__(self):
        return len(self.X_load)
        
    def __getitem__(self, idx):
        # The 4th item was previously calendar info, we mock an empty tensor for legacy compat if needed
        return self.X_load[idx], self.X_cal[idx], self.Y_load[idx], torch.zeros(1)
