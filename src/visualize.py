"""
CAEG-Net Visualization Suite
Generates publication-quality charts for load forecasting comparison,
gating weight allocation heatmaps, metric comparison bar charts, and ablation results.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

# Set clean aesthetic plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "savefig.bbox": "tight"})

def plot_forecast_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: str = "outputs/forecast_comparison.png",
    sample_indices: List[int] = [10, 85, 180]
):
    """
    Plots 24-hour load forecasts for sample test windows across different operating regimes.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 4 * len(sample_indices)))
    
    if len(sample_indices) == 1:
        axes = [axes]
        
    models_to_plot = ["CAEG-Net", "Standard MoE", "Static Ensemble", "LSTM Expert", "TCN Expert", "CNN Expert"]
    colors = {
        "Ground Truth": "#1f77b4",
        "CAEG-Net": "#d62728",        # Bright Red for our model
        "Standard MoE": "#ff7f0e",   # Orange
        "Static Ensemble": "#2ca02c",# Green
        "LSTM Expert": "#9467bd",   # Purple
        "TCN Expert": "#8c564b",    # Brown
        "CNN Expert": "#e377c2"     # Pink
    }
    
    hours = np.arange(1, 25)
    
    for ax_idx, sample_idx in enumerate(sample_indices):
        ax = axes[ax_idx]
        
        # Ground truth load
        gt = results_dict["CAEG-Net"]["raw_targets"][sample_idx]
        ax.plot(hours, gt, label="Ground Truth Load", color=colors["Ground Truth"], linewidth=2.5, linestyle="-", marker="o", markersize=4)
        
        for m_name in models_to_plot:
            if m_name in results_dict:
                pred = results_dict[m_name]["preds_unscaled"][sample_idx]
                lw = 2.2 if m_name == "CAEG-Net" else 1.4
                ls = "-" if m_name == "CAEG-Net" else "--"
                ax.plot(hours, pred, label=m_name, color=colors.get(m_name, "gray"), linewidth=lw, linestyle=ls)
                
        ax.set_title(f"24-Hour Horizon Load Forecast (Sample Window {sample_idx})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Forecast Horizon (Hours)")
        ax.set_ylabel("Electricity Load (kW)")
        ax.set_xticks(hours)
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved forecast comparison plot to {output_path}")


def plot_expert_gating_heatmap(
    expert_weights: np.ndarray,
    raw_loads: np.ndarray,
    output_path: str = "outputs/expert_gating_heatmap.png",
    num_samples: int = 120
):
    """
    Plots dynamic allocation of expert weights (LSTM, TCN, CNN) over consecutive time steps
    alongside the historical load curve to demonstrate context adaptation.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 1.2]}, sharex=True)
    
    steps = np.arange(num_samples)
    load_slice = raw_loads[:num_samples, 0]  # First hour forecast of each window
    
    # Top plot: Load profile
    ax1.plot(steps, load_slice, color="#1f77b4", linewidth=1.8, label="Actual Load")
    ax1.set_ylabel("Load (kW)", fontsize=11, fontweight="bold")
    ax1.set_title("CAEG-Net Dynamic Expert Selection vs Operating Conditions", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # Bottom plot: Expert Weights Stacked Area Plot
    weights_slice = expert_weights[:num_samples].T  # (3, num_samples)
    w_lstm, w_tcn, w_cnn = weights_slice[0], weights_slice[1], weights_slice[2]
    
    ax2.stackplot(
        steps,
        w_lstm, w_tcn, w_cnn,
        labels=["LSTM Expert Weight", "TCN Expert Weight", "CNN Expert Weight"],
        colors=["#9467bd", "#8c564b", "#e377c2"],
        alpha=0.85
    )
    ax2.set_ylabel("Expert Softmax Weight", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Consecutive Sliding Window Index (Hours)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper right", frameon=True, facecolor="white")
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved expert gating heatmap plot to {output_path}")


def plot_metric_bar_chart(
    metrics_summary: Dict[str, Dict[str, float]],
    output_path: str = "outputs/metrics_comparison.png"
):
    """
    Renders comparative bar charts for MAE, RMSE, MAPE, and R2 across all evaluated models.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(metrics_summary.keys())
    
    mae_vals = [metrics_summary[m]["mae"] for m in models]
    rmse_vals = [metrics_summary[m]["rmse"] for m in models]
    mape_vals = [metrics_summary[m]["mape"] for m in models]
    r2_vals = [metrics_summary[m]["r2"] for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors highlighting CAEG-Net
    bar_colors = ["#d62728" if "CAEG" in m else "#4c72b0" for m in models]
    
    # Subplot 1: MAE
    axes[0, 0].barh(models, mae_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[0, 0].set_title("Mean Absolute Error (MAE - Lower is Better)", fontweight="bold")
    axes[0, 0].set_xlabel("MAE (kW)")
    for i, v in enumerate(mae_vals):
        axes[0, 0].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=10, fontweight="bold" if "CAEG" in models[i] else "normal")
        
    # Subplot 2: RMSE
    axes[0, 1].barh(models, rmse_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[0, 1].set_title("Root Mean Squared Error (RMSE - Lower is Better)", fontweight="bold")
    axes[0, 1].set_xlabel("RMSE (kW)")
    for i, v in enumerate(rmse_vals):
        axes[0, 1].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=10, fontweight="bold" if "CAEG" in models[i] else "normal")
        
    # Subplot 3: MAPE
    axes[1, 0].barh(models, mape_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1, 0].set_title("Mean Absolute Percentage Error (MAPE % - Lower is Better)", fontweight="bold")
    axes[1, 0].set_xlabel("MAPE (%)")
    for i, v in enumerate(mape_vals):
        axes[1, 0].text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=10, fontweight="bold" if "CAEG" in models[i] else "normal")
        
    # Subplot 4: R2 Score
    axes[1, 1].barh(models, r2_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1, 1].set_title("Coefficient of Determination (R² Score - Higher is Better)", fontweight="bold")
    axes[1, 1].set_xlabel("R² Score")
    for i, v in enumerate(r2_vals):
        axes[1, 1].text(v - 0.04, i, f"{v:.4f}", va="center", fontsize=10, fontweight="bold" if "CAEG" in models[i] else "normal")
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved metrics bar chart to {output_path}")


def plot_ablation_results(
    ablation_summary: Dict[str, Dict[str, float]],
    output_path: str = "outputs/ablation_study_chart.png"
):
    """
    Renders bar chart specifically highlighting the ablation study findings:
    1. Standard MoE (Raw input gate)
    2. CAEG-Net (Context Gate w/o Closed Loop)
    3. CAEG-Net Full (Context Gate w/ Closed-Loop Error Feedback)
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(ablation_summary.keys())
    
    mape_vals = [ablation_summary[m]["mape"] for m in models]
    rmse_vals = [ablation_summary[m]["rmse"] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, mape_vals, width, label="MAPE (%)", color="#ff7f0e", edgecolor="black")
    rects2 = ax.bar(x + width/2, rmse_vals, width, label="RMSE (kW)", color="#1f77b4", edgecolor="black")
    
    ax.set_ylabel("Error Metric Value", fontweight="bold")
    ax.set_title("Ablation Study: Impact of Context Representation & Closed-Loop Error Feedback", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.legend(loc="upper right")
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ablation study plot to {output_path}")
