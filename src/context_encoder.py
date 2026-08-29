"""
CAEG-Net Explicit Context Encoder
Extracts interpretable meta-features (Trend, Volatility, Periodicity, Recent Error)
from historical load input sequences to drive expert gating decisions.
"""

import torch
import torch.nn as nn
from typing import Tuple

class ContextEncoder(nn.Module):
    """
    Explicit Context Encoder for CAEG-Net.
    
    Inputs:
        x_load: Batch of historical load sequences, shape (B, T, 1) or (B, T)
        recent_error: Batch of recent prediction errors (closed-loop feedback), shape (B, 1) or scalar
        
    Outputs:
        context_vector: (B, 4) containing:
            1. Trend Strength (c1)
            2. Volatility Level (c2)
            3. Periodicity Strength (c3, 24h autocorrelation)
            4. Recent Prediction Error (c4, closed-loop feedback signal)
    """
    def __init__(self, lookback_hours: int = 168, eps: float = 1e-6):
        super().__init__()
        self.lookback_hours = lookback_hours
        self.eps = eps
        
        # Learnable adaptive scaling weights for combining normalized features
        self.context_scale = nn.Parameter(torch.ones(4))
        
    def compute_trend(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates relative trend strength comparing short-term (last 24h) 
        vs full window (168h) moving average.
        x shape: (B, T)
        Returns: (B, 1)
        """
        ma_short = x[:, -24:].mean(dim=1, keepdim=True)
        ma_long = x.mean(dim=1, keepdim=True)
        trend = (ma_short - ma_long) / (torch.abs(ma_long) + self.eps)
        return trend

    def compute_volatility(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates normalized volatility level using relative standard deviation 
        of first-order differences.
        x shape: (B, T)
        Returns: (B, 1)
        """
        diffs = x[:, 1:] - x[:, :-1]
        std_diff = torch.std(diffs, dim=1, keepdim=True, unbiased=False)
        mean_x = torch.abs(torch.mean(x, dim=1, keepdim=True)) + self.eps
        volatility = std_diff / mean_x
        return volatility

    def compute_periodicity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates 24-hour lag autocorrelation (periodicity strength).
        x shape: (B, T)
        Returns: (B, 1)
        """
        mean_x = x.mean(dim=1, keepdim=True)
        x_centered = x - mean_x
        
        # Lag 24 correlation
        x_t = x_centered[:, 24:]
        x_t_lag = x_centered[:, :-24]
        
        numerator = torch.sum(x_t * x_t_lag, dim=1, keepdim=True)
        denominator = torch.sum(x_centered ** 2, dim=1, keepdim=True) + self.eps
        periodicity = numerator / denominator
        return periodicity

    def forward(
        self, 
        x_load: torch.Tensor, 
        recent_error: torch.Tensor
    ) -> torch.Tensor:
        """
        x_load: (B, T) or (B, T, 1) - historical load values
        recent_error: (B, 1) - recent model prediction error (closed loop signal)
        """
        if x_load.dim() == 3:
            x_load = x_load.squeeze(-1)
            
        trend = self.compute_trend(x_load)           # (B, 1)
        volatility = self.compute_volatility(x_load) # (B, 1)
        periodicity = self.compute_periodicity(x_load) # (B, 1)
        
        if recent_error is None:
            recent_error = torch.zeros_like(trend)
        elif recent_error.dim() == 1:
            recent_error = recent_error.unsqueeze(-1)
            
        # Stack into explicit 4D context vector (B, 4)
        raw_context = torch.cat([trend, volatility, periodicity, recent_error], dim=1)
        
        # LayerNorm to ensure all 4 meta-features are on identical unit scale
        norm_context = torch.nn.functional.layer_norm(raw_context, normalized_shape=(4,))
        
        # Apply learnable feature scaling
        context_vector = norm_context * self.context_scale
        return context_vector
