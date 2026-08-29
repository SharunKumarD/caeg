"""
CAEG-Net Evaluation & Ablation Suite
Executes model evaluation on test sets, computes metrics (MAE, RMSE, MAPE, R2, Params, Speed),
and runs structured ablation experiments.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List, Tuple

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    """Computes Mean Absolute Percentage Error (%)"""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)

def count_parameters(model: nn.Module) -> int:
    """Counts total trainable parameters in PyTorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model_on_dataset(
    model: nn.Module,
    test_loader: DataLoader,
    scaler,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates PyTorch forecasting model on test DataLoader.
    Inverse-transforms scaled predictions back to actual kW/MW electricity load scale
    to calculate real-world metrics.
    """
    model.eval()
    model = model.to(device)
    
    all_preds_scaled = []
    all_targets_scaled = []
    all_raw_targets = []
    all_expert_weights = []
    
    recent_error_state = torch.zeros((test_loader.batch_size, 1), device=device)
    
    inference_times = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_seq = batch["x_seq"].to(device)
            y_seq = batch["y_seq"].to(device)
            raw_x_load = batch["raw_x_load"].to(device)
            raw_y = batch["raw_y"].numpy()
            
            if recent_error_state.size(0) != x_seq.size(0):
                recent_error_state = torch.zeros((x_seq.size(0), 1), device=device)
                
            t_start = time.time()
            fused_pred, expert_weights, _ = model(
                x_seq=x_seq,
                raw_x_load=raw_x_load,
                recent_error=recent_error_state
            )
            t_end = time.time()
            inference_times.append((t_end - t_start) * 1000.0)  # ms
            
            all_preds_scaled.append(fused_pred.cpu().numpy())
            all_targets_scaled.append(y_seq.cpu().numpy())
            all_raw_targets.append(raw_y)
            all_expert_weights.append(expert_weights.cpu().numpy())
            
            # Closed loop error update for next test step
            batch_mae = torch.mean(torch.abs(fused_pred - y_seq), dim=1, keepdim=True)
            recent_error_state = 0.8 * recent_error_state + 0.2 * batch_mae

    preds_scaled = np.concatenate(all_preds_scaled, axis=0)  # (N, forecast_hours)
    targets_scaled = np.concatenate(all_targets_scaled, axis=0)
    raw_targets = np.concatenate(all_raw_targets, axis=0)
    expert_weights = np.concatenate(all_expert_weights, axis=0)  # (N, num_experts)
    
    # Inverse scaling to actual physical load values
    # Feature 0 is load in our StandardScaler
    mean_load = scaler.mean_[0]
    scale_load = scaler.scale_[0]
    
    preds_unscaled = preds_scaled * scale_load + mean_load
    
    # Metrics calculation
    mae = mean_absolute_error(raw_targets, preds_unscaled)
    mse = mean_squared_error(raw_targets, preds_unscaled)
    rmse = float(np.sqrt(mse))
    mape = compute_mape(raw_targets, preds_unscaled)
    r2 = r2_score(raw_targets.ravel(), preds_unscaled.ravel())
    
    param_count = count_parameters(model)
    avg_inference_latency_ms = float(np.mean(inference_times))
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "param_count": param_count,
        "inference_latency_ms": avg_inference_latency_ms,
        "preds_unscaled": preds_unscaled,
        "raw_targets": raw_targets,
        "expert_weights": expert_weights
    }
