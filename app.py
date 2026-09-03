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
st.set_page_config(page_title="CAEG-Net — PJM Electricity Load Forecasting", layout="wide")

st.title("CAEG-Net")
st.subheader("Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting")
st.markdown("**Dataset:** Modern PJM")

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

# Application layout
st.markdown("### About CAEG-Net")
st.markdown("""
- **LSTM Expert:** Captures long-term sequential dependencies.
- **TCN Expert:** Extracts multi-scale temporal patterns.
- **CNN Expert:** Identifies local anomalies and sudden spikes.
- **Context Encoder:** Evaluates current trend, volatility, and periodicity.
- **Dynamic Expert Gating:** Intelligently combines expert predictions based on context.
- **Closed-Loop Error Feedback:** Feeds previous prediction errors back into the context to correct trajectory drifts.
""")

st.markdown("---")
st.markdown("### PJM Load Forecasting")

model = load_model()
X_test, Y_test, scaler = load_and_prepare_data()

if X_test is not None and model is not None:
    # Run full sequential inference once and cache it
    preds, w, ctx = run_full_inference(model, X_test, Y_test)
    
    # Calculate total metrics
    st.markdown("### Global Model Performance (Test Set)")
    metrics = evaluate_all(preds, Y_test)
    cols = st.columns(4)
    cols[0].metric("MSE", f"{metrics['MSE']:.4f}")
    cols[1].metric("MAE", f"{metrics['MAE']:.4f}")
    cols[2].metric("RMSE", f"{metrics['RMSE']:.4f}")
    cols[3].metric("R²", f"{metrics['R2']:.4f}")
    
    st.markdown("---")
    st.markdown("### Prediction Explorer")
    
    sample_idx = st.slider("Select Test Sample Index", 0, len(X_test) - 1, 0)
    
    if st.button("Predict Load"):
        actual = Y_test[sample_idx].numpy().flatten()
        prediction = preds[sample_idx].numpy().flatten()
        
        # Inverse transform to original scale
        actual_unscaled = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
        pred_unscaled = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        
        st.success(f"Generated 24-hour prediction for sample {sample_idx}.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(actual_unscaled, label="Actual Load", marker='o')
            ax.plot(pred_unscaled, label="Predicted Load", marker='x')
            ax.set_title("24-Hour Load Forecast")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Load (MW)")
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            st.markdown("#### Dynamic Expert Gating Weights")
            weights = w[sample_idx].numpy().flatten()
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            experts = ["LSTM", "TCN", "CNN"]
            ax2.bar(experts, weights, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax2.set_title("Expert Contributions")
            ax2.set_ylabel("Weight")
            ax2.set_ylim(0, 1)
            st.pyplot(fig2)
            
        st.markdown("#### Context Features & Closed-Loop Feedback")
        context_vals = ctx[sample_idx].numpy().flatten()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trend Strength", f"{context_vals[0]:.4f}")
        c2.metric("Volatility Level", f"{context_vals[1]:.4f}")
        c3.metric("Periodicity", f"{context_vals[2]:.4f}")
        
        # For error feedback, the model uses `prev_err` in its forward pass.
        # We can calculate the mean of the absolute previous error used.
        if sample_idx == 0:
            prev_err_mae = 0.0
        else:
            prev_err_unscaled = scaler.inverse_transform( (preds[sample_idx-1] - Y_test[sample_idx-1]).numpy().reshape(-1,1) )
            prev_err_mae = np.mean(np.abs(prev_err_unscaled))
            
        c4.metric("Closed-Loop Feedback (Prev MAE)", f"{prev_err_mae:.2f} MW")

else:
    st.warning("Could not load data or model. Please check the repository paths.")
