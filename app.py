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

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### CAEG-Net")
    st.markdown("**Context-Adaptive Expert Gating Network**")
    st.markdown("---")
    st.markdown("**Dataset:** Modern PJM")
    st.markdown("**Input Window:** 168 hours")
    st.markdown("**Forecast Horizon:** 24 hours")
    st.markdown("**Model:** CAEG-Net")
    st.markdown("**Experts:** LSTM / TCN / CNN")
    st.markdown("---")


# ================= MAIN HERO =================
st.title("CAEG-Net")
st.subheader("Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting")
st.markdown("A deep learning architecture that intelligently routes multi-scale temporal data to specialized experts (LSTM, TCN, CNN) based on current context (Trend, Volatility, Periodicity) and closed-loop prediction errors.")
st.markdown("---")

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
    # Cache inference in session state to prevent CPU throttling
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
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("MSE", f"{metrics['MSE']:.4f}")
    mc2.metric("MAE", f"{metrics['MAE']:.4f}")
    mc3.metric("RMSE", f"{metrics['RMSE']:.4f}")
    mc4.metric("R²", f"{metrics['R2']:.4f}")
    
    st.markdown("---")
    
    # ================= PREDICTION EXPLORER =================
    st.markdown("### Prediction Explorer")
    
    sample_idx = st.slider("Select Test Sample Index", 0, len(X_test) - 1, 0)
    
    if st.button("Generate 24-Hour Forecast", type="primary"):
        actual = Y_test[sample_idx].numpy().flatten()
        prediction = preds[sample_idx].numpy().flatten()
        
        # Inverse transform to original scale
        actual_unscaled = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
        pred_unscaled = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        
        # ================= FORECAST SUMMARY =================
        st.markdown("#### Forecast Summary")
        s1, s2, s3 = st.columns(3)
        s1.metric("Peak Predicted Load", f"{pred_unscaled.max():.2f} MW", f"Hour {pred_unscaled.argmax()}")
        s2.metric("Minimum Predicted Load", f"{pred_unscaled.min():.2f} MW", f"Hour {pred_unscaled.argmin()}")
        s3.metric("Average Predicted Load", f"{pred_unscaled.mean():.2f} MW")
        
        # ================= VISUALIZATIONS =================
        col_plot, col_gating = st.columns([2, 1])
        
        with col_plot:
            st.markdown("#### Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(actual_unscaled, label="Actual Load", marker='o', linewidth=2)
            ax.plot(pred_unscaled, label="Predicted Load", marker='x', linewidth=2)
            ax.set_title(f"24-Hour Load Forecast (Sample {sample_idx})")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Load (MW)")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)
            
        with col_gating:
            st.markdown("#### Expert Gating Weights")
            weights = w[sample_idx].numpy().flatten()
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            experts = ["LSTM", "TCN", "CNN"]
            bars = ax2.bar(experts, weights, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax2.set_title("Expert Contributions")
            ax2.set_ylabel("Weight")
            ax2.set_ylim(0, 1)
            # Add numerical labels
            for bar in bars:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
            st.pyplot(fig2)
            
        # ================= CONTEXT ANALYSIS =================
        st.markdown("#### Context Analysis")
        context_vals = ctx[sample_idx].numpy().flatten()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trend Strength", f"{context_vals[0]:.4f}")
        c2.metric("Volatility Level", f"{context_vals[1]:.4f}")
        c3.metric("Periodicity", f"{context_vals[2]:.4f}")
        
        # Correct closed-loop error unscaling
        if sample_idx == 0:
            prev_err_unscaled_mae = 0.0
        else:
            prev_err = (preds[sample_idx-1] - Y_test[sample_idx-1]).numpy().flatten()
            # Error is a magnitude, not an absolute value to be offset by the mean.
            # Convert scaled error magnitude to original MW magnitude using scaler.scale_
            scale_factor = scaler.scale_[0]
            prev_err_unscaled_mae = np.mean(np.abs(prev_err)) * scale_factor
            
        c4.metric("Closed-Loop Feedback (Prev MAE)", f"{prev_err_unscaled_mae:.2f} MW")
        
        # ================= FORECAST TABLE =================
        with st.expander("View 24-Hour Tabular Data"):
            error_unscaled = np.abs(pred_unscaled - actual_unscaled)
            df_table = pd.DataFrame({
                "Hour": range(24),
                "Actual Load (MW)": actual_unscaled,
                "Predicted Load (MW)": pred_unscaled,
                "Absolute Error (MW)": error_unscaled
            })
            st.dataframe(df_table.style.format("{:.2f}"))

else:
    st.warning("Could not load data or model. Please check the repository paths.")

# ================= MODEL INFORMATION =================
st.markdown("---")
st.markdown("### Architecture Details")
col_arch1, col_arch2 = st.columns(2)
with col_arch1:
    st.info("**LSTM Expert:** Long-term temporal dependencies")
    st.info("**TCN Expert:** Multi-scale temporal patterns")
    st.info("**CNN Expert:** Local patterns and sudden variations")
with col_arch2:
    st.info("**Context Encoder:** Trend, volatility, periodicity and feedback")
    st.info("**Dynamic Expert Gating:** Adaptive expert combination")
    st.info("**Closed-Loop Feedback:** Uses previous prediction error")
