"""
CAEG-Net Gating Networks Module
Implements:
1. StandardGatingNetwork (Baseline raw input router)
2. CAEGGatingNetwork (Context-Adaptive router with explicit meta-features & closed-loop feedback)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardGatingNetwork(nn.Module):
    """
    Standard MoE Gating Network (Baseline).
    Takes raw input sequence features directly into an MLP to generate expert softmax weights.
    Input shape: (B, T, num_features)
    Output shape: (B, num_experts) -> softmax weights summing to 1
    """
    def __init__(
        self,
        lookback_hours: int = 168,
        num_features: int = 7,
        num_experts: int = 3,
        hidden_dim: int = 64
    ):
        super().__init__()
        flat_dim = lookback_hours * num_features
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts)
        )
        
    def forward(self, x_seq: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # x_seq shape: (B, T, C) -> Flatten to (B, T*C)
        b = x_seq.size(0)
        x_flat = x_seq.reshape(b, -1)
        logits = self.mlp(x_flat)
        weights = F.softmax(logits, dim=-1)
        return weights


class CAEGGatingNetwork(nn.Module):
    """
    Context-Adaptive Expert Gating Network (CAEG-Net - Our Proposed Gate).
    Routes between experts based on explicit 4D context vector:
    c = [Trend Strength, Volatility Level, Periodicity Strength, Recent Error Feedback]
    Input shape: (B, 4)
    Output shape: (B, num_experts) -> dynamic expert softmax weights [alpha_LSTM, alpha_TCN, alpha_CNN]
    """
    def __init__(
        self,
        context_dim: int = 4,
        num_experts: int = 3,
        hidden_dim: int = 32,
        temperature: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts)
        )
        
    def forward(self, context_vector: torch.Tensor) -> torch.Tensor:
        # context_vector shape: (B, 4)
        logits = self.gate_mlp(context_vector) / self.temperature
        weights = F.softmax(logits, dim=-1)
        return weights
