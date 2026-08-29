import math
import torch
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import CAEGNet modules from the package
from model import CAEGNet
from metrics import evaluate_all

app = FastAPI(title="CAEG-Net API", description="Electricity Load Forecasting Backend")

# 1. Enable CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os

models = {}

@app.on_event("startup")
async def startup_event():
    print(f"Starting up... Loading CAEG-Net models on {device}")
    
    datasets = {
        "ELEC2": "caeg_net_elec2_best.pth",
        "Modern": "caeg_net_modern_best.pth"
    }
    
    for ds_name, weight_path in datasets.items():
        m = CAEGNet(input_dim=1, seq_len=168, pred_len=24).to(device)
        if os.path.exists(weight_path):
            print(f"Found trained weights for {ds_name} at {weight_path}. Loading...")
            m.load_state_dict(torch.load(weight_path, map_location=device))
        else:
            print(f"No trained weights found for {ds_name}. Using initialized weights.")
        m.eval()
        models[ds_name] = m
        
    print("Models loaded successfully.")

# 2. Pydantic Models for Request/Response
class PredictRequest(BaseModel):
    dataset_type: str = "ELEC2"
    historical_load: Optional[List[float]] = None
    calendar_features: Optional[List[List[float]]] = None
    previous_prediction_error: Optional[List[float]] = None
    scenario: Optional[str] = None # Helper to auto-generate specific scenarios

class PredictResponse(BaseModel):
    forecast_24h: List[float]
    actual_next_24h: Optional[List[float]]
    gating_weights: Dict[str, float]
    context_features: Dict[str, float]
    expert_predictions: Dict[str, List[float]]
    metrics: Optional[Dict[str, float]]
    benchmark_table: Optional[List[Dict[str, Any]]]

# Helper function to generate synthetic data if payload is empty
def generate_synthetic_data(scenario: str, seq_len: int = 168, pred_len: int = 24):
    total_len = seq_len + pred_len
    time = np.arange(total_len)
    
    # Base pattern: Daily + Weekly
    load = 100 * np.sin(2 * np.pi * time / 24) + 50 * np.sin(2 * np.pi * time / (24*7)) + 300
    
    if scenario == "Weekday Peak Load":
        load += 50 * np.sin(2 * np.pi * time / 24) # Amplify daily peak
    elif scenario == "Weekend Off-Peak":
        load -= 100 # Lower overall load
    elif scenario == "Extreme Weather / Volatile Spike":
        # Add a massive spike in the last 24 hours of history
        spike_start = seq_len - 24
        load[spike_start:seq_len] += 200 * np.random.rand(24)
    elif scenario == "Holiday Cycle":
        load = 80 * np.sin(2 * np.pi * time / 24) + 250 # Flatter curve
        
    # Add random noise
    load += np.random.normal(0, 10, total_len)
    
    # Standardize (approximate)
    mean, std = np.mean(load), np.std(load)
    load_scaled = (load - mean) / (std + 1e-8)
    
    history = load_scaled[:seq_len].tolist()
    future_actual = load_scaled[seq_len:].tolist()
    
    return history, future_actual

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    seq_len = 168
    pred_len = 24
    
    # Select dynamic model
    ds_type = req.dataset_type if req.dataset_type in models else "ELEC2"
    active_model = models[ds_type]
    
    # 3. Handle empty/partial payload (auto-sample)
    actual_next = None
    if not req.historical_load or len(req.historical_load) != seq_len:
        scenario = req.scenario if req.scenario else "Weekday Peak Load"
        hist, actual_next = generate_synthetic_data(scenario, seq_len, pred_len)
    else:
        hist = req.historical_load
    
    if req.previous_prediction_error and len(req.previous_prediction_error) == pred_len:
        prev_error = req.previous_prediction_error
    else:
        prev_error = [0.0] * pred_len

    # Prepare tensors
    x_load = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    x_err = torch.tensor(prev_error, dtype=torch.float32).unsqueeze(0).to(device)

    # 4. Inference execution
    with torch.no_grad():
        out_lstm = active_model.expert_lstm(x_load)
        out_tcn = active_model.expert_tcn(x_load)
        out_cnn = active_model.expert_cnn(x_load)
        
        context_emb = active_model.context_encoder(x_load, x_err)
        weights = active_model.gating_network(context_emb)
        
        # We also need to apply the correct residual feedback logic
        final_out, _ = active_model(x_load, x_err)
        
    # Extract Context Features manually for the API response
    trend_val = (x_load[0, -1, 0] - x_load[0, 0, 0]).item()
    vol_val = x_load[0, :, 0].std().item()
    
    lag = min(24, seq_len // 2)
    part1 = x_load[0, lag:, 0].unsqueeze(0)
    part2 = x_load[0, :-lag, 0].unsqueeze(0)
    periodicity_val = torch.cosine_similarity(part1, part2, dim=1).item()
    
    recent_err_val = x_err[0].abs().mean().item()

    # Calculate metrics if ground truth is available (auto-generated)
    metrics_res = None
    if actual_next is not None:
        y_true = torch.tensor(actual_next, dtype=torch.float32).unsqueeze(0).to(device)
        metrics_res = evaluate_all(final_out, y_true)

    # 4.5 Load Benchmark Results
    import pandas as pd
    benchmark_data = []
    csv_file = f"benchmark_results_{ds_type.lower()}.csv"
    if ds_type == "Modern":
        csv_file = "benchmark_results_modern.csv"
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        benchmark_data = df.to_dict(orient="records")

    # 5. Construct JSON response
    response = PredictResponse(
        forecast_24h=final_out[0].tolist(),
        actual_next_24h=actual_next,
        gating_weights={
            "LSTM": float(weights[0, 0]),
            "TCN": float(weights[0, 1]),
            "CNN": float(weights[0, 2])
        },
        context_features={
            "trend_strength": trend_val,
            "volatility_level": vol_val,
            "periodicity_strength": periodicity_val,
            "recent_error": recent_err_val
        },
        expert_predictions={
            "LSTM": out_lstm[0].tolist(),
            "TCN": out_tcn[0].tolist(),
            "CNN": out_cnn[0].tolist()
        },
        metrics=metrics_res,
        benchmark_table=benchmark_data
    )
    
    return response

@app.get("/sample-scenarios")
async def get_sample_scenarios():
    """
    Returns predefined test scenarios so the frontend can easily switch contexts 
    and visualize how the gating network shifts expert weights.
    """
    return {
        "scenarios": [
            {
                "id": "weekday_peak",
                "name": "Weekday Peak Load",
                "description": "High baseline with strong daily periodicity. Expect LSTM/TCN to dominate."
            },
            {
                "id": "weekend_offpeak",
                "name": "Weekend Off-Peak",
                "description": "Lower baseline, stable pattern."
            },
            {
                "id": "extreme_weather",
                "name": "Extreme Weather / Volatile Spike",
                "description": "High volatility and a sudden spike at the end. Expect CNN to activate for local feature extraction."
            },
            {
                "id": "holiday_cycle",
                "name": "Holiday Cycle",
                "description": "Flatter curve with disrupted normal periodicity."
            }
        ]
    }

# Mount static frontend files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
