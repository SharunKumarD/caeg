import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# CAEG-Net imports
from model import CAEGNet
from dataset import load_real_data
from metrics import evaluate_all

# Page configuration
st.set_page_config(page_title="CAEG-Net — PJM Load Forecasting", layout="wide", initial_sidebar_state="expanded")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    /* Main Layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hero Title */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #000;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4C3C92;
        margin-bottom: 1rem;
    }
    .hero-desc {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 2rem;
        max-width: 800px;
        line-height: 1.6;
    }

    /* Architecture Flowchart */
    .arch-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        overflow-x: auto;
    }
    .arch-box {
        display: flex;
        flex-direction: column;
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        min-width: 150px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .arch-box-title {
        font-weight: bold;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .arch-box-desc {
        font-size: 0.75rem;
        color: #64748b;
        line-height: 1.3;
    }
    .arch-arrow {
        color: #94a3b8;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 0 10px;
    }
    
    /* Colored Titles */
    .t-lstm { color: #2563eb; }
    .t-tcn { color: #16a34a; }
    .t-cnn { color: #7c3aed; }
    .t-ctx { color: #d97706; }
    .t-gate { color: #dc2626; }
    .t-feed { color: #059669; }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚡ CAEG-Net")
    st.caption("Context-Adaptive Expert Gating Network")
    st.markdown("---")
    
    st.markdown("### MODEL INFO")
    st.markdown("🗄️ **Dataset**\nModern PJM")
    st.markdown("⏱️ **Input Window**\n168 hours")
    st.markdown("📅 **Forecast Horizon**\n24 hours")
    st.markdown("🧠 **Model**\nCAEG-Net")
    st.markdown("👥 **Experts**\nLSTM / TCN / CNN")
    st.markdown("---")
    
    st.markdown("### SYSTEM STATUS")
    st.markdown("✅ Model Loaded")
    st.markdown("✅ Data Loaded")
    st.markdown("✅ Inference Ready")
    
    st.markdown("---")
    st.caption("College ML Project Demonstration")
    st.caption("🔒 Inference Only Model")

# ================= MAIN HERO =================
st.markdown('<div class="hero-title">CAEG-Net</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-desc">CAEG-Net intelligently combines the strengths of LSTM, TCN, and CNN experts through a context-aware gating mechanism. The model adapts dynamically based on trend, volatility, periodicity, and closed-loop prediction errors to produce accurate 24-hour load forecasts.</div>', unsafe_allow_html=True)


# ================= MODEL ARCHITECTURE =================
st.markdown("### Model Architecture Overview")
arch_html = """
<div class="arch-container">
    <div class="arch-box">
        <div class="arch-box-title t-lstm">🧠 LSTM Expert</div>
        <div class="arch-box-desc">Captures long-term<br>temporal dependencies</div>
    </div>
    <div class="arch-arrow">></div>
    <div class="arch-box">
        <div class="arch-box-title t-tcn">📈 TCN Expert</div>
        <div class="arch-box-desc">Extracts multi-scale<br>temporal patterns</div>
    </div>
    <div class="arch-arrow">></div>
    <div class="arch-box">
        <div class="arch-box-title t-cnn">💠 CNN Expert</div>
        <div class="arch-box-desc">Identifies local patterns<br>and sudden spikes</div>
    </div>
    <div class="arch-arrow">></div>
    <div class="arch-box">
        <div class="arch-box-title t-ctx">📊 Context Encoder</div>
        <div class="arch-box-desc">Trend, volatility,<br>periodicity, feedback</div>
    </div>
    <div class="arch-arrow">></div>
    <div class="arch-box">
        <div class="arch-box-title t-gate">🔗 Dynamic Gating</div>
        <div class="arch-box-desc">Adaptive combination<br>of expert predictions</div>
    </div>
    <div class="arch-arrow">></div>
    <div class="arch-box">
        <div class="arch-box-title t-feed">🔄 Closed-Loop Feedback</div>
        <div class="arch-box-desc">Uses previous errors to<br>correct future forecasts</div>
    </div>
</div>
"""
st.markdown(arch_html, unsafe_allow_html=True)

# ================= INITIALIZATION & DATA LOADING =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 168
pred_len = 24
csv_path = Path("data") / "dataset_modern.csv"
checkpoint_path = Path("checkpoints") / "caeg_net_modern_best.pth"

@st.cache_resource
def load_model():
    model = CAEGNet(input_dim=1, seq_len=seq_len, pred_len=pred_len).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        st.error(f"Checkpoint not found at {checkpoint_path}")
    model.eval()
    return model

@st.cache_data
def load_and_prepare_data():
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}")
        return None, None, None
        
    X_load, X_cal, Y_load, scaler = load_real_data(csv_path=str(csv_path), seq_len=seq_len, pred_len=pred_len, train_ratio=0.8)
    
    total_samples = len(X_load)
    train_size = int(0.8 * total_samples)
    
    X_test_load = X_load[train_size:]
    Y_test_load = Y_load[train_size:]
    
    return X_test_load, Y_test_load, scaler

def run_full_inference(_model, X_test, Y_test):
    _model.eval()
    preds_list, w_list, ctx_list = [], [], []
    num_samples = len(X_test)
    
    with torch.no_grad():
        x = X_test.to(device)
        y = Y_test.to(device)
        prev_err = torch.zeros(1, pred_len).to(device)
        
        for i in range(num_samples):
            pred, w_i = _model(x[i:i+1], prev_err)
            
            # Manually calculate context features exactly as ContextEncoder does
            trend = (x[i:i+1, -1, 0] - x[i:i+1, 0, 0]).unsqueeze(-1)
            volatility = x[i:i+1, :, 0].std(dim=1).unsqueeze(-1)
            lag = min(24, 168 // 2)
            periodicity = torch.cosine_similarity(x[i:i+1, lag:, 0], x[i:i+1, :-lag, 0], dim=1, eps=1e-8).unsqueeze(-1)
            recent_error = torch.tanh(prev_err.abs().mean(dim=1, keepdim=True))
            ctx_i = torch.cat([trend, volatility, periodicity, recent_error], dim=1)
            
            preds_list.append(pred.cpu())
            w_list.append(w_i.cpu())
            ctx_list.append(ctx_i.cpu())
            prev_err = (pred - y[i:i+1]).detach()
            
        preds = torch.cat(preds_list, dim=0)
        w = torch.cat(w_list, dim=0)
        ctx = torch.cat(ctx_list, dim=0)
        
    return preds, w, ctx

model = load_model()
X_test, Y_test, scaler = load_and_prepare_data()

if X_test is not None and model is not None:
    # ================= PERFORMANCE OPTIMIZATION =================
    if "inference_done" not in st.session_state:
        with st.spinner("Initializing sequential closed-loop inference..."):
            preds, w, ctx = run_full_inference(model, X_test, Y_test)
            st.session_state["preds"] = preds
            st.session_state["w"] = w
            st.session_state["ctx"] = ctx
            st.session_state["inference_done"] = True
    else:
        preds = st.session_state["preds"]
        w = st.session_state["w"]
        ctx = st.session_state["ctx"]
        
    # ================= MODEL PERFORMANCE =================
    st.markdown("### Model Performance (Test Set)")
    metrics = evaluate_all(preds, Y_test)
    
    # Custom colored metrics layout
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("📉 MSE", f"{metrics['MSE']:.4f}", "Lower is better", delta_color="inverse")
    with mc2:
        st.metric("📉 MAE", f"{metrics['MAE']:.4f}", "Lower is better", delta_color="inverse")
    with mc3:
        st.metric("📉 RMSE", f"{metrics['RMSE']:.4f}", "Lower is better", delta_color="inverse")
    with mc4:
        st.metric("⭐ R² Score", f"{metrics['R2']:.4f}", "Higher is better", delta_color="normal")
    
    st.markdown("---")
    
    # ================= MIDDLE SECTION =================
    col_explorer, col_plot, col_gating = st.columns([1, 2.5, 1.5])
    
    with col_explorer:
        st.markdown("#### Prediction Explorer")
        sample_idx = st.slider("Select Test Sample Index", 0, len(X_test) - 1, 0, label_visibility="collapsed")
        st.markdown(f"**Selected Sample:** {sample_idx} / {len(X_test)-1}")
        
        actual = Y_test[sample_idx].numpy().flatten()
        prediction = preds[sample_idx].numpy().flatten()
        
        # Inverse transform to original scale
        actual_unscaled = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
        pred_unscaled = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

    with col_plot:
        st.markdown("#### Actual vs Predicted Load (24-Hour Forecast)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actual_unscaled, label="Actual Load", marker='o', markersize=4, color='#2563eb', linewidth=2)
        ax.plot(pred_unscaled, label="Predicted Load", marker='o', markersize=4, color='#dc2626', linewidth=2)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Load (MW)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Custom legend styling
        ax.legend(frameon=False, loc="upper left")
        
        # Clean axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
        
    with col_gating:
        st.markdown("#### Dynamic Expert Gating Weights")
        weights = w[sample_idx].numpy().flatten()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        experts = ["LSTM", "TCN", "CNN"]
        colors = ['#2563eb', '#dc2626', '#16a34a']
        bars = ax2.bar(experts, weights, color=colors, width=0.6)
        ax2.set_ylabel("Weight")
        ax2.set_ylim(0, 1.1)
        
        # Add numerical labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.03, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
            
        # Clean axes
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)
        
    st.markdown("---")
    
    # ================= BOTTOM SECTION =================
    col_context, col_summary, col_table = st.columns([1.5, 1, 1.2])
    
    with col_context:
        st.markdown("#### Context Features & Closed-Loop Feedback")
        context_vals = ctx[sample_idx].numpy().flatten()
        
        # Correct closed-loop error unscaling
        if sample_idx == 0:
            prev_err_unscaled_mae = 0.0
        else:
            prev_err = (preds[sample_idx-1] - Y_test[sample_idx-1]).numpy().flatten()
            scale_factor = scaler.scale_[0]
            prev_err_unscaled_mae = np.mean(np.abs(prev_err)) * scale_factor
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trend Strength", f"{context_vals[0]:.4f}")
        c2.metric("Volatility Level", f"{context_vals[1]:.4f}")
        c3.metric("Periodicity", f"{context_vals[2]:.4f}")
        c4.metric("Feedback (Prev MAE)", f"{prev_err_unscaled_mae:.2f} MW")
        
        st.info("ℹ️ CAEG-Net uses sequential closed-loop inference; each prediction error is fed back to the model for the next time step, ensuring robust short-term load forecasting.")
        
    with col_summary:
        st.markdown("#### Forecast Summary")
        st.metric("↗️ Peak Predicted", f"{pred_unscaled.max():.2f} MW", f"Hour {pred_unscaled.argmax()}", delta_color="off")
        st.metric("↘️ Minimum Predicted", f"{pred_unscaled.min():.2f} MW", f"Hour {pred_unscaled.argmin()}", delta_color="off")
        st.metric("➡️ Average Predicted", f"{pred_unscaled.mean():.2f} MW", delta_color="off")
        
    with col_table:
        st.markdown("#### 24-Hour Forecast Table")
        error_unscaled = pred_unscaled - actual_unscaled
        df_table = pd.DataFrame({
            "Hour": range(24),
            "Actual Load (MW)": actual_unscaled,
            "Predicted Load (MW)": pred_unscaled,
            "Error (MW)": error_unscaled
        })
        st.dataframe(df_table.style.format("{:.2f}"), height=250, use_container_width=True)

else:
    st.warning("Could not load data or model. Please check the repository paths.")
