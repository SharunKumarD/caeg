"""
CAEG-Net Quick Demo Script
Runs a fast 5-epoch training & evaluation pipeline to verify system execution,
generate sample outputs, and demonstrate dynamic expert routing.
"""

import os
import torch
import pandas as pd
import numpy as np

from src.dataset import generate_synthetic_stlf_data, create_stlf_dataloaders
from src.model import CAEGNet, StandardMoEModel, StaticEnsembleModel
from src.experts import LSTMExpert, TCNExpert, CNNExpert
from src.train import train_model
from src.evaluate import evaluate_model_on_dataset
from src.visualize import (
    plot_forecast_comparison,
    plot_expert_gating_heatmap,
    plot_metric_bar_chart,
    plot_ablation_results
)

class SingleExpertWrapper(torch.nn.Module):
    def __init__(self, expert_model):
        super().__init__()
        self.expert = expert_model
    def forward(self, x_seq, raw_x_load=None, recent_error=None):
        pred = self.expert(x_seq)
        b = x_seq.size(0)
        device = x_seq.device
        dummy_weights = torch.zeros((b, 3), device=device)
        return pred, dummy_weights, {}

def main():
    print("=" * 70)
    print(" CAEG-Net Quick Verification & Demo Execution")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running on device: {device}")
    
    output_dir = os.path.abspath("outputs")
    ckpt_dir = os.path.abspath("checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("\n[1/4] Generating Synthetic Hourly Load Data (90 days for fast demo)...")
    df = generate_synthetic_stlf_data(num_days=90, random_seed=42)
    
    train_loader, val_loader, test_loader, scaler = create_stlf_dataloaders(
        df, lookback_hours=168, forecast_hours=24, batch_size=32
    )
    
    num_features = len(test_loader.dataset.feature_cols)
    
    models = {
        "Single LSTM": SingleExpertWrapper(LSTMExpert(input_size=num_features, forecast_hours=24)),
        "Single TCN": SingleExpertWrapper(TCNExpert(input_size=num_features, forecast_hours=24)),
        "Single CNN": SingleExpertWrapper(CNNExpert(input_size=num_features, forecast_hours=24)),
        "Static Ensemble": StaticEnsembleModel(lookback_hours=168, forecast_hours=24, num_features=num_features),
        "Standard MoE": StandardMoEModel(lookback_hours=168, forecast_hours=24, num_features=num_features),
        "CAEG-Net (No Closed Loop)": CAEGNet(lookback_hours=168, forecast_hours=24, num_features=num_features, use_closed_loop_error=False),
        "CAEG-Net": CAEGNet(lookback_hours=168, forecast_hours=24, num_features=num_features, use_closed_loop_error=True)
    }
    
    epochs = 10
    eval_results = {}
    
    print(f"\n[2/4] Training Models for {epochs} Epochs...")
    for name, model in models.items():
        sanitized_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=1e-3,
            device=device,
            model_name=sanitized_name,
            save_dir=ckpt_dir
        )
        
        res = evaluate_model_on_dataset(model, test_loader, scaler, device=device)
        res["training_time_sec"] = history["total_training_time"]
        eval_results[name] = res
        print(f"  -> {name:<26} | MAE: {res['mae']:6.2f} kW | RMSE: {res['rmse']:6.2f} kW | MAPE: {res['mape']:5.2f}% | R²: {res['r2']:.4f}")

    print("\n[3/4] Generating Benchmark Summary Table...")
    rows = []
    for name, res in eval_results.items():
        rows.append({
            "Model": name,
            "MAE (kW)": f"{res['mae']:.2f}",
            "RMSE (kW)": f"{res['rmse']:.2f}",
            "MAPE (%)": f"{res['mape']:.2f}%",
            "R² Score": f"{res['r2']:.4f}",
            "Params": f"{res['param_count']:,}",
            "Latency (ms)": f"{res['inference_latency_ms']:.2f}ms"
        })
    demo_df = pd.DataFrame(rows)
    print("\n" + demo_df.to_string(index=False))
    
    print("\n[4/4] Rendering Charts to outputs/...")
    plot_forecast_comparison(eval_results, output_path=os.path.join(output_dir, "forecast_comparison.png"))
    plot_expert_gating_heatmap(
        eval_results["CAEG-Net"]["expert_weights"],
        eval_results["CAEG-Net"]["raw_targets"],
        output_path=os.path.join(output_dir, "expert_gating_heatmap.png")
    )
    
    metrics_summary = {k: {"mae": v["mae"], "rmse": v["rmse"], "mape": v["mape"], "r2": v["r2"]} for k, v in eval_results.items()}
    plot_metric_bar_chart(metrics_summary, output_path=os.path.join(output_dir, "metrics_comparison.png"))
    
    ablation_keys = ["Standard MoE", "CAEG-Net (No Closed Loop)", "CAEG-Net"]
    ablation_summary = {k: metrics_summary[k] for k in ablation_keys if k in metrics_summary}
    plot_ablation_results(ablation_summary, output_path=os.path.join(output_dir, "ablation_study_chart.png"))
    
    print("\n[+] Demo Execution Finished Successfully! All output plots and reports generated in outputs/.")

if __name__ == "__main__":
    main()
