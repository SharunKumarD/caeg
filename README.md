# CAEG-Net: A Context-Adaptive Expert Gating Network with Closed-Loop Error Feedback for Short-Term Electricity Load Forecasting

## Team Members
* Sharun Kumar D
* Vinay Viswanathan
* Tabrez N

---

## Overview
CAEG-Net is a state-of-the-art ensemble learning architecture designed to tackle the highly volatile, non-stationary nature of short-term electricity load forecasting. Rather than relying on a single monolith architecture, CAEG-Net employs a Mixture-of-Experts (MoE) approach dynamically guided by explicitly engineered contextual time-series features.

## Key Features
* **Explicit Context Encoder**: Extracts macro-level time-series attributes (Trend Strength, Volatility Level, and Periodicity Strength) to understand the current grid regime.
* **Dynamic Expert Routing**: A lightweight MLP Gating Network evaluates the context embedding to dynamically allocate percentage weights across three specialized experts:
  * **LSTM Expert**: Captures long-term sequential dependencies.
  * **TCN Expert**: Leverages causal dilated convolutions for efficient temporal modeling.
  * **CNN Expert**: Extracts highly localized spatial features in response to sudden volatile spikes.
* **Closed-Loop Error Feedback**: Uniquely feeds the `t-1` prediction error back into the context encoder at step `t`, allowing the network to rapidly identify and correct systemic bias in real-time.

---

## Repository Structure
* `model.py` / `experts.py` / `context.py`: PyTorch implementations of the core CAEG-Net architecture.
* `dataset.py`: Data pipeline generating robust, multi-regime load forecasting sequences.
* `main.py`: Full Early-Stopping Time-Series training loop using AdamW and L2 Regularization.
* `ablation.py`: Automated benchmarking script comparing CAEG-Net against 5 ablation baselines.
* `app.py`: FastAPI production backend.
* `static/`: HTML/CSS/JS frontend application featuring live predictions, Chart.js visualizations, and Explainable AI contextual breakdowns.

---

## How to Run

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the early-stopping training pipeline to generate the optimal model weights (`caeg_net_best.pth`).
```bash
python main.py
```

### 3. Run the Ablation Study
Evaluate the trained model against the static baselines. This generates `benchmark_results.csv`, `benchmark_results.json`, and a visualization plot `benchmark_comparison.png`.
```bash
python ablation.py
```

### 4. Launch the Web Dashboard
Start the unified FastAPI server to serve both the API and the interactive frontend operations dashboard.
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Once the server is running, open your browser and navigate to: [http://localhost:8000](http://localhost:8000)
