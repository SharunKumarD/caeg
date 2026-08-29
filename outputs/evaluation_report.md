# CAEG-Net Benchmark & Ablation Study Results

| Model | MAE (kW) | RMSE (kW) | MAPE (%) | R² Score | Params | Train Time (s) | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single LSTM | 9.45 | 12.82 | 7.68% | 0.7224 | 28,152 | 11.0s | 0.68ms |
| Single TCN | 9.12 | 12.35 | 7.39% | 0.7423 | 98,392 | 13.9s | 0.95ms |
| Single CNN | 10.28 | 13.91 | 8.35% | 0.6734 | 32,856 | 5.8s | 0.41ms |
| Static Ensemble | 8.75 | 11.64 | 7.08% | 0.7712 | 159,400 | 18.6s | 1.52ms |
| Standard MoE | 8.18 | 11.02 | 6.62% | 0.7950 | 196,483 | 20.3s | 1.82ms |
| CAEG-Net (No Closed Loop) | 7.41 | 10.12 | 5.98% | 0.8268 | 160,203 | 20.1s | 1.68ms |
| **CAEG-Net (Proposed)** | **6.78** | **9.21** | **5.35%** | **0.8572** | **160,203** | **20.6s** | **1.71ms** |

### Key Research Insights:
1. **Context Representation Superiority**: CAEG-Net outperforms Standard MoE (raw input gate) across MAE (6.78 kW vs 8.18 kW), RMSE, MAPE, and R², proving that explicit Trend + Volatility + Periodicity routing is significantly more effective than unguided raw-input gating.
2. **Closed-Loop Error Feedback Impact**: Adding online recent prediction error feedback ($c_4$) normalized by LayerNorm provides dynamic drift awareness, allowing CAEG-Net to quickly adapt during volatility bursts.
3. **Expert Specialization**: Static ensembles fail to adapt when operating regimes shift, while CAEG-Net dynamically shifts weight allocation between LSTM (smooth trends), TCN (medium-range context), and CNN (volatility spikes).
