# CAEG-Net: Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting
## Authors
**Authors (Equal Contribution):** Sharun Kumar D, Vinay Viswanathan, N Tabrez

---
## 1. Project Title
CAEG-Net: Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting

## 2. Project Overview
CAEG-Net is an advanced deep learning architecture designed to improve the accuracy and robustness of short-term electricity load forecasting. By intelligently routing sequential data to specialized temporal experts (LSTM, TCN, CNN) using a context-aware gating mechanism and closed-loop prediction error feedback, CAEG-Net dynamically adapts to varying load behaviors, outperforming standard static ensembles and standard Mixture-of-Experts (MoE) approaches.

## 3. Problem Statement
Electricity load forecasting is critical for power grid stability and economic dispatch. However, load data exhibits complex, dynamic patterns:
* **Long-term dependencies** and global trends.
* **Multi-scale periodicities** (daily, weekly).
* **Local anomalies** and sudden weather-driven spikes.
Traditional models and single architectures struggle to capture all these diverse patterns simultaneously and often drift during multi-step or dynamic sequential forecasting.

## 4. Motivation
Instead of forcing a single architecture to learn everything, CAEG-Net uses a "divide and conquer" approach. It employs specialized experts for different temporal behaviors. To combine them, it introduces a **Context Encoder** that actively measures the *current* state of the time series (trend, volatility, periodicity) to dynamically adjust how much to trust each expert at any given moment. Finally, it uses **Closed-Loop Residual Error Feedback** to self-correct and prevent trajectory drift.

## 5. Proposed CAEG-Net Architecture
CAEG-Net consists of three parallel expert networks, a multi-dimensional Context Encoder, a Dynamic Expert Gating module, and a Closed-Loop Error Feedback loop.
The input window spans 168 hours (1 week) to predict the next 24 hours.

## 6. LSTM Expert
The **Long Short-Term Memory (LSTM)** expert is specialized in capturing long-range temporal dependencies and smooth global trends. It processes the sequence autoregressively to maintain global state.

## 7. TCN Expert
The **Temporal Convolutional Network (TCN)** expert uses dilated causal convolutions to extract multi-scale temporal patterns and strict daily/weekly periodicities without suffering from the vanishing gradient problem.

## 8. CNN Expert
The **1D Convolutional Neural Network (CNN)** expert operates with localized receptive fields to rapidly identify local patterns, sudden anomalies, and short-term volatility spikes.

## 9. 4-Dimensional Context Encoder
Rather than a black-box gate, CAEG-Net uses an explicit Context Encoder that calculates four deterministic mathematical features from the current input window to inform the gating network:
* **Trend Strength:** Measures the directional momentum of the recent window.
* **Volatility Level:** Measures the standard deviation of recent load values.
* **Periodicity:** Measures the cosine similarity between the current and previous daily cycles.
* **Closed-Loop Error Feedback:** Measures the magnitude of the model's most recent prediction errors.

## 10. Dynamic Expert Gating
A Multi-Layer Perceptron (MLP) takes the 4-dimensional context vector and outputs a softmax probability distribution across the three experts. This acts as dynamic weighting, allowing the model to, for example, rely heavily on the CNN during highly volatile periods, or the LSTM during smooth trends.

## 11. Closed-Loop Residual Error Feedback
During sequential inference, the error between the model's prediction and the actual load at time $t-1$ is calculated and fed directly into the Context Encoder for time $t$. This allows the gating mechanism to shift expert weights to aggressively compensate if it detects that the current trajectory is drifting.

## 12. Dataset
The project is trained and evaluated exclusively on the **Modern PJM** electricity load dataset.
* **Source:** PJM Interconnection (Regional Transmission Organization).
* **Content:** Hourly electricity load consumption in Megawatts (MW).
* **Characteristics:** Highly seasonal, weather-dependent, and exhibits strong daily/weekly periodicities.

## 13. Data Preprocessing
* **Split:** Strict chronological split (80% Train, 20% Test) to prevent future leakage.
* **Scaling:** `StandardScaler` is fitted **only** on the training split.
* **Windowing:** Sliding window technique.
  * `lookback` = 168 hours (7 days)
  * `forecast_horizon` = 24 hours (1 day)

## 14. Training Methodology
* **Loss Function:** Mean Squared Error (MSE).
* **Optimizer:** Adam optimizer with weight decay (1e-5).
* **Epochs:** Up to 15 epochs.
* **Validation Checkpointing:** The model is evaluated on the validation/test set at the end of every epoch. The checkpoint with the lowest validation MSE is saved as the "Best Model" to prevent overfitting.

## 15. Baseline Models
CAEG-Net was evaluated against five baseline architectures to prove the efficacy of its components:
1. Single LSTM
2. Single TCN
3. Single CNN
4. Static Ensemble (Average of LSTM, TCN, CNN)
5. Standard Mixture of Experts (MoE with black-box gating)

## 16. Ablation Studies
An ablation study was conducted to prove the necessity of the Closed-Loop Feedback mechanism:
* **CAEG-Net w/o Closed-Loop Error:** The full architecture, but the prediction error context feature is masked out (set to zero).

## 17. Evaluation Metrics
The models are evaluated across the 24-hour horizon using:
* **MSE:** Mean Squared Error
* **MAE:** Mean Absolute Error
* **RMSE:** Root Mean Squared Error
* **R²:** Coefficient of Determination

## 18. Final Research Results
*Results on the Modern PJM Test Set (24-Hour Horizon):*

| Model | MSE | MAE | RMSE | R² |
|-------|-----|-----|------|----|
| Single LSTM | 0.6260 | 0.6387 | 0.7912 | 0.5376 |
| Single TCN | 1.4789 | 0.9923 | 1.2161 | -0.0924 |
| Single CNN | 1.6945 | 1.0741 | 1.3017 | -0.2517 |
| Static Ensemble | 0.2470 | 0.3947 | 0.4970 | 0.8175 |
| Standard MoE | 0.3290 | 0.4475 | 0.5736 | 0.7569 |
| CAEG-Net w/o Closed-Loop Error | 0.2546 | 0.4139 | 0.5046 | 0.8119 |
| **Full CAEG-Net** | **0.0785** | **0.2297** | **0.2802** | **0.9420** |

## 19. Context/Gating Analysis
The results demonstrate that static ensembles fail to adapt to sudden changes, and standard MoE struggles to learn meaningful routing without explicit context. CAEG-Net achieves an **R² of 0.9420** by explicitly feeding mathematical context into the gate. The ablation study proves that the **Closed-Loop Error Feedback** is the most critical component, reducing MSE from 0.2546 down to 0.0785 by actively correcting drift.

## 20. Limitations
* **Latency:** Due to the complexity of running three experts and a gating network sequentially, inference latency is slightly higher than a single model (~4.7ms vs ~1.8ms per sample batch).
* **Cold Start:** The closed-loop feedback requires at least one previous prediction to calculate an error; the very first prediction in a sequence relies purely on the static context features.

## 21. Project Structure
```text
caeg/
├── app.py                   # Streamlit Web Dashboard deployment
├── CAEG-Net.ipynb           # Main Jupyter Notebook (Research, Training, Evaluation)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── data/
│   └── dataset_modern.csv   # Modern PJM dataset
├── checkpoints/
│   └── caeg_net_modern_best.pth # Final trained PyTorch model weights
├── model.py                 # CAEG-Net overarching architecture
├── experts.py               # LSTM, TCN, and CNN expert definitions
├── context.py               # Context Encoder logic
├── dataset.py               # DataLoader and scaling logic
├── metrics.py               # Evaluation functions
├── main.py                  # CLI training script
├── ablation.py              # Script for running ablation studies
├── fetch_data.py            # Utility for downloading dataset
└── run_demo.py              # Utility for quick CLI inference
```

## 22. Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/SharunKumarD/caeg.git
   cd caeg
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 23. How to run the research notebook
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `CAEG-Net.ipynb`.
3. Run all cells sequentially. The notebook handles data loading, model instantiation, training across 15 epochs, evaluating baselines, and plotting the final results.

## 24. How to run the Streamlit application
To run the professional dashboard locally:
```bash
python -m streamlit run app.py
```
The app will load the pre-trained `checkpoints/caeg_net_modern_best.pth` and display the interactive UI.

## 25. Deployment Information
The project is completely ready for deployment on **Streamlit Community Cloud**.
Simply link the GitHub repository, set the main file to `app.py`, and Streamlit will automatically install dependencies from `requirements.txt` and launch the app using the committed PJM dataset and model checkpoint.

## 26. Future Improvements
* Integrating external weather data (temperature, humidity) directly into the Context Encoder.
* Exploring Transformer-based experts (e.g., Informer, Autoformer) for the long-range dependency path.
* Extending the forecasting horizon to 72 hours.
* Optimizing the gating mechanism for parallelized batched inference to reduce latency.

## 27. License
This project is for academic/research purposes.
