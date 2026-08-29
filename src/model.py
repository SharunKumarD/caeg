"""
CAEG-Net Core Architecture & Baseline Model Wrappers
Integrates Context Encoder, Experts, Gating, and Dynamic Weighted Fusion.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .context_encoder import ContextEncoder
from .experts import LSTMExpert, TCNExpert, CNNExpert
from .gating import StandardGatingNetwork, CAEGGatingNetwork

class CAEGNet(nn.Module):
    """
    Proposed Context-Adaptive Expert Gating Network (CAEG-Net).
    Combines:
    - Context Encoder (Meta-features: Trend, Volatility, Periodicity, Recent Error)
    - LSTM, TCN, CNN Experts
    - CAEG Gating Network
    - Dynamic Weighted Fusion
    """
    def __init__(
        self,
        lookback_hours: int = 168,
        forecast_hours: int = 24,
        num_features: int = 7,
        use_closed_loop_error: bool = True
    ):
        super().__init__()
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.use_closed_loop_error = use_closed_loop_error
        
        # 1. Context Encoder
        self.context_encoder = ContextEncoder(lookback_hours=lookback_hours)
        
        # 2. Experts Pool
        self.lstm_expert = LSTMExpert(input_size=num_features, forecast_hours=forecast_hours)
        self.tcn_expert = TCNExpert(input_size=num_features, forecast_hours=forecast_hours)
        self.cnn_expert = CNNExpert(input_size=num_features, forecast_hours=forecast_hours)
        
        # 3. Gating Network
        self.gating_net = CAEGGatingNetwork(context_dim=4, num_experts=3)
        
    def forward(
        self, 
        x_seq: torch.Tensor, 
        raw_x_load: torch.Tensor, 
        recent_error: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inputs:
            x_seq: Scaled multi-feature sequence, shape (B, T, num_features)
            raw_x_load: Raw unscaled historical load sequence, shape (B, T)
            recent_error: Closed-loop error feedback signal, shape (B, 1)
            
        Returns:
            fused_forecast: (B, forecast_hours)
            expert_weights: (B, 3) [alpha_LSTM, alpha_TCN, alpha_CNN]
            expert_forecasts: Dict containing individual expert outputs
        """
        if not self.use_closed_loop_error:
            recent_error = torch.zeros_like(raw_x_load[:, :1])
            
        # Extract explicit context vector (B, 4)
        context_vector = self.context_encoder(raw_x_load, recent_error)
        
        # Get dynamic expert weights from context (B, 3)
        expert_weights = self.gating_net(context_vector)
        
        # Generate expert predictions (B, forecast_hours)
        pred_lstm = self.lstm_expert(x_seq)
        pred_tcn = self.tcn_expert(x_seq)
        pred_cnn = self.cnn_expert(x_seq)
        
        # Dynamic Weighted Fusion
        # Stack predictions to (B, 3, forecast_hours)
        stacked_preds = torch.stack([pred_lstm, pred_tcn, pred_cnn], dim=1)
        
        # Expand weights for matrix multiplication (B, 3, 1)
        w_expanded = expert_weights.unsqueeze(-1)
        
        # Weighted sum across expert dimension
        fused_forecast = torch.sum(stacked_preds * w_expanded, dim=1)
        
        expert_dict = {
            "lstm": pred_lstm,
            "tcn": pred_tcn,
            "cnn": pred_cnn,
            "context_vector": context_vector
        }
        
        return fused_forecast, expert_weights, expert_dict


class StandardMoEModel(nn.Module):
    """
    Standard Baseline MoE Model using raw feature input gating.
    """
    def __init__(
        self,
        lookback_hours: int = 168,
        forecast_hours: int = 24,
        num_features: int = 7
    ):
        super().__init__()
        self.lstm_expert = LSTMExpert(input_size=num_features, forecast_hours=forecast_hours)
        self.tcn_expert = TCNExpert(input_size=num_features, forecast_hours=forecast_hours)
        self.cnn_expert = CNNExpert(input_size=num_features, forecast_hours=forecast_hours)
        
        self.gating_net = StandardGatingNetwork(
            lookback_hours=lookback_hours,
            num_features=num_features,
            num_experts=3
        )
        
    def forward(self, x_seq: torch.Tensor, raw_x_load: torch.Tensor = None, recent_error: torch.Tensor = None):
        expert_weights = self.gating_net(x_seq)
        
        pred_lstm = self.lstm_expert(x_seq)
        pred_tcn = self.tcn_expert(x_seq)
        pred_cnn = self.cnn_expert(x_seq)
        
        stacked_preds = torch.stack([pred_lstm, pred_tcn, pred_cnn], dim=1)
        w_expanded = expert_weights.unsqueeze(-1)
        fused_forecast = torch.sum(stacked_preds * w_expanded, dim=1)
        
        return fused_forecast, expert_weights, {"lstm": pred_lstm, "tcn": pred_tcn, "cnn": pred_cnn}


class StaticEnsembleModel(nn.Module):
    """
    Static Ensemble Baseline (Equal fixed weighting: 1/3, 1/3, 1/3).
    """
    def __init__(
        self,
        lookback_hours: int = 168,
        forecast_hours: int = 24,
        num_features: int = 7
    ):
        super().__init__()
        self.lstm_expert = LSTMExpert(input_size=num_features, forecast_hours=forecast_hours)
        self.tcn_expert = TCNExpert(input_size=num_features, forecast_hours=forecast_hours)
        self.cnn_expert = CNNExpert(input_size=num_features, forecast_hours=forecast_hours)
        
    def forward(self, x_seq: torch.Tensor, raw_x_load: torch.Tensor = None, recent_error: torch.Tensor = None):
        pred_lstm = self.lstm_expert(x_seq)
        pred_tcn = self.tcn_expert(x_seq)
        pred_cnn = self.cnn_expert(x_seq)
        
        fused_forecast = (pred_lstm + pred_tcn + pred_cnn) / 3.0
        
        b = x_seq.size(0)
        device = x_seq.device
        fixed_weights = torch.full((b, 3), 1.0/3.0, device=device)
        
        return fused_forecast, fixed_weights, {"lstm": pred_lstm, "tcn": pred_tcn, "cnn": pred_cnn}
