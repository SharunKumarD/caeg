"""
CAEG-Net Dataset Module
High-fidelity electricity load dataset generator & PyTorch Dataset pipeline for Short-Term Electricity Load Forecasting (STLF).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional

def generate_synthetic_stlf_data(
    num_days: int = 365,
    sampling_rate_hours: int = 1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic hourly electricity load time series data with multi-seasonality,
    regime shifts (weekdays vs weekends, peak vs off-peak), temperature sensitivity,
    holiday spikes, and sudden volatility bursts.
    """
    np.random.seed(random_seed)
    total_hours = num_days * 24
    time_idx = pd.date_range(start="2024-01-01", periods=total_hours, freq="h")
    
    # Base daily pattern (24h period) - morning and evening peak
    hour_of_day = time_idx.hour.values
    daily_pattern = (
        120.0 
        + 35.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        + 15.0 * np.sin(4 * np.pi * (hour_of_day - 17) / 24)
    )
    
    # Weekly pattern (168h period) - lower demand on weekends
    day_of_week = time_idx.dayofweek.values
    is_weekend = (day_of_week >= 5).astype(float)
    weekly_factor = 1.0 - 0.18 * is_weekend
    
    # Seasonal temperature simulation (annual cycle)
    day_of_year = time_idx.dayofyear.values
    temp_annual = 20.0 + 15.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0)
    temp_daily = 5.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    temperature = temp_annual + temp_daily + np.random.normal(0, 2.0, size=total_hours)
    
    # Non-linear temperature load response (HVAC demand for extreme cold or heat)
    cooling_heating_load = 2.5 * np.maximum(0, temperature - 22.0)**1.2 + 3.0 * np.maximum(0, 10.0 - temperature)**1.1
    
    # Macro Trend (slow long-term economic / structural shift)
    trend = 5.0 * np.arange(total_hours) / (365.0 * 24.0)
    
    # Volatility regime shifts (random extreme weather / industrial spikes)
    volatility_mask = np.zeros(total_hours)
    # Inject volatility bursts on random days (approx 10% of time)
    num_spikes = int(num_days * 0.12)
    spike_days = np.random.choice(np.arange(1, num_days - 1), size=num_spikes, replace=False)
    for d in spike_days:
        start_h = d * 24 + np.random.randint(10, 18)
        duration = np.random.randint(2, 6)
        volatility_mask[start_h : start_h + duration] = np.random.uniform(40.0, 90.0, size=duration)
    
    # Gaussian noise
    noise = np.random.normal(0, 4.0, size=total_hours)
    
    # Total Load calculation
    load = (daily_pattern * weekly_factor) + cooling_heating_load + trend + volatility_mask + noise
    load = np.maximum( load, 30.0) # Ensure strictly positive load
    
    df = pd.DataFrame({
        "timestamp": time_idx,
        "load": load,
        "temperature": temperature,
        "hour": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour_sin": np.sin(2 * np.pi * hour_of_day / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour_of_day / 24.0),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7.0),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7.0),
        "volatility_regime": (volatility_mask > 0).astype(float)
    })
    
    return df

class STLFDataset(Dataset):
    """
    PyTorch Dataset for Short-Term Electricity Load Forecasting with sliding windows.
    Input window size: lookback_hours (default 168 = 7 days)
    Output horizon: forecast_hours (default 24 = 1 day)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        lookback_hours: int = 168,
        forecast_hours: int = 24,
        scaler: Optional[StandardScaler] = None,
        is_train: bool = True
    ):
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.is_train = is_train
        
        # Feature columns used for input sequence
        self.feature_cols = [
            "load", "temperature", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"
        ]
        self.target_col_idx = 0  # "load" is index 0
        
        feature_data = df[self.feature_cols].values
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(feature_data)
        else:
            self.scaler = scaler
            self.scaled_data = self.scaler.transform(feature_data)
            
        self.raw_load = df["load"].values
        self.timestamps = df["timestamp"].values
        
        # Number of samples
        self.total_steps = len(df) - lookback_hours - forecast_hours + 1
        
    def __len__(self) -> int:
        return max(0, self.total_steps)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Input sequence: shape (lookback_hours, num_features)
        x_seq = self.scaled_data[idx : idx + self.lookback_hours]
        
        # Target sequence (Load only): shape (forecast_hours,)
        y_seq = self.scaled_data[
            idx + self.lookback_hours : idx + self.lookback_hours + self.forecast_hours,
            self.target_col_idx
        ]
        
        # Unscaled raw target load for evaluation & context error calculation
        raw_y = self.raw_load[
            idx + self.lookback_hours : idx + self.lookback_hours + self.forecast_hours
        ]
        raw_x_load = self.raw_load[idx : idx + self.lookback_hours]
        
        return {
            "x_seq": torch.tensor(x_seq, dtype=torch.float32),
            "y_seq": torch.tensor(y_seq, dtype=torch.float32),
            "raw_x_load": torch.tensor(raw_x_load, dtype=torch.float32),
            "raw_y": torch.tensor(raw_y, dtype=torch.float32),
            "idx": idx
        }

def create_stlf_dataloaders(
    df: pd.DataFrame,
    lookback_hours: int = 168,
    forecast_hours: int = 24,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Splits time series chronologically into Train, Validation, and Test sets,
    and returns PyTorch DataLoaders along with the fitted scaler.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end - lookback_hours : val_end].reset_index(drop=True)
    test_df = df.iloc[val_end - lookback_hours :].reset_index(drop=True)
    
    train_dataset = STLFDataset(train_df, lookback_hours, forecast_hours, scaler=None, is_train=True)
    scaler = train_dataset.scaler
    
    val_dataset = STLFDataset(val_df, lookback_hours, forecast_hours, scaler=scaler, is_train=False)
    test_dataset = STLFDataset(test_df, lookback_hours, forecast_hours, scaler=scaler, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, scaler
