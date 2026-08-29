import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CAEGNet
from experts import LSTMExpert, TCNExpert, CNNExpert
from metrics import evaluate_all
from dataset import load_real_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

class SingleExpertWrapper(nn.Module):
    def __init__(self, expert_model):
        super().__init__()
        self.model = expert_model
    def forward(self, x, prev_error=None):
        return self.model(x), None

class StaticEnsemble(nn.Module):
    def __init__(self, m1, m2, m3):
        super().__init__()
        self.m1, self.m2, self.m3 = m1, m2, m3
    def forward(self, x, prev_error=None):
        out = (self.m1(x) + self.m2(x) + self.m3(x)) / 3.0
        return out, None

class StandardMoE(nn.Module):
    def __init__(self, m1, m2, m3, input_dim=1, seq_len=168):
        super().__init__()
        self.m1, self.m2, self.m3 = m1, m2, m3
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
    def forward(self, x, prev_error=None):
        w = self.gate(x)
        out = w[:, 0:1] * self.m1(x) + w[:, 1:2] * self.m2(x) + w[:, 2:3] * self.m3(x)
        return out, w

def evaluate_variant(name, model, X_test, Y_test, use_closed_loop=False):
    import time
    print(f"Evaluating: {name} ", end="", flush=True)
    model.eval()
    model.to(device)
    
    predictions = []
    prev_err = torch.zeros((1, 24), device=device)
    
    start_time = time.time()
    with torch.no_grad():
        for i in range(len(X_test)):
            pred, _ = model(X_test[i:i+1].to(device), prev_err)
            predictions.append(pred.cpu().numpy())
            
            if use_closed_loop:
                prev_err = (pred - Y_test[i:i+1].to(device)).detach()
            else:
                prev_err = torch.zeros((1, 24), device=device)
                
    latency = (time.time() - start_time) / len(X_test) * 1000
    print("... Done.")
    
    preds_tensor = torch.tensor(np.concatenate(predictions, axis=0), dtype=torch.float32)
    actuals_tensor = Y_test.clone().detach().to(torch.float32)
    
    metrics = evaluate_all(preds_tensor, actuals_tensor)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics["Parameters"] = num_params
    metrics["Latency (ms)"] = latency
    
    return metrics

def run_evaluation_on_dataset(ds_name, csv_path, weight_path, out_csv):
    print(f"\n{'='*80}")
    print(f"   Running CAEG-Net Benchmark on {ds_name} Dataset   ")
    print(f"{'='*80}")
    
    seq_len = 168
    pred_len = 24
    X_load, X_cal, Y_load, scaler = load_real_data(csv_path=csv_path, seq_len=seq_len, pred_len=pred_len)
    
    total_samples = len(X_load)
    train_size = int(0.8 * total_samples)
    X_test, Y_test = X_load[train_size:], Y_load[train_size:]
    
    lstm = LSTMExpert(1, 64, 2, pred_len)
    tcn = TCNExpert(1, [32, 64, 64], 3, pred_len)
    cnn = CNNExpert(1, 64, pred_len)
    
    full_caeg = CAEGNet(1, seq_len, pred_len)
    if os.path.exists(weight_path):
        print(f"Loading trained weights from {weight_path}...")
        full_caeg.load_state_dict(torch.load(weight_path, map_location=device))
        lstm.load_state_dict(full_caeg.expert_lstm.state_dict())
        tcn.load_state_dict(full_caeg.expert_tcn.state_dict())
        cnn.load_state_dict(full_caeg.expert_cnn.state_dict())
    else:
        print(f"{weight_path} not found. Using randomly initialized weights.")
        
    variants = [
        ("Baseline 1 (Single LSTM)", SingleExpertWrapper(lstm), False),
        ("Baseline 2 (Single TCN)", SingleExpertWrapper(tcn), False),
        ("Baseline 3 (Single CNN)", SingleExpertWrapper(cnn), False),
        ("Baseline 4 (Static Ensemble)", StaticEnsemble(lstm, tcn, cnn), False),
        ("Baseline 5 (Standard MoE)", StandardMoE(lstm, tcn, cnn), False),
        ("Ablation 1 (CAEG-Net w/o Closed-Loop Error)", full_caeg, False),
        ("Proposed Model (Full CAEG-Net)", full_caeg, True),
    ]
    
    results = []
    for name, v_model, use_loop in variants:
        res = evaluate_variant(name, v_model, X_test, Y_test, use_closed_loop=use_loop)
        res["Model"] = name
        results.append(res)
        
    df = pd.DataFrame(results)
    cols = ["Model", "MSE", "MAE", "RMSE", "R2", "Parameters", "Latency (ms)"]
    df = df[cols]
    
    df.to_csv(out_csv, index=False)
    return df

def main():
    datasets = [
        ("ELEC2", "data/dataset_elec2.csv", "caeg_net_elec2_best.pth", "benchmark_results_elec2.csv"),
        ("Modern PJM", "data/dataset_modern.csv", "caeg_net_modern_best.pth", "benchmark_results_modern.csv")
    ]
    
    all_dfs = {}
    for ds_name, csv_path, weight_path, out_csv in datasets:
        df = run_evaluation_on_dataset(ds_name, csv_path, weight_path, out_csv)
        all_dfs[ds_name] = df

    print("\n\n### Dual Dataset Benchmark Results")
    
    for ds_name in all_dfs:
        print(f"\n#### Dataset: {ds_name}")
        print(all_dfs[ds_name].to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    main()
