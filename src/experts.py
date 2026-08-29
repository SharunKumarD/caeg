"""
CAEG-Net Expert Models
Implements three functionally complementary expert networks:
1. LSTM Expert (Long-range sequential memory)
2. TCN Expert (Dilated causal temporal convolutional receptive field)
3. CNN Expert (Local pattern & volatility spike feature extractor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMExpert(nn.Module):
    """
    LSTM Expert: 2-layer LSTM for capturing long-term sequential temporal dependencies.
    Input shape: (B, T, num_features)
    Output shape: (B, forecast_hours)
    """
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        forecast_hours: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, forecast_hours)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take last time step hidden state
        last_hidden = lstm_out[:, -1, :]
        out = self.head(last_hidden)
        return out


class ChokuCausalConv1d(nn.Module):
    """Causal 1D Convolution with specified dilation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalResidualBlock(nn.Module):
    """TCN Residual Block with Causal Convolutions, BatchNorm, and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ChokuCausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = ChokuCausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return F.relu(out + res)


class TCNExpert(nn.Module):
    """
    Temporal Convolutional Network (TCN) Expert.
    Uses dilated causal 1D convolutions (dilations 1, 2, 4, 8) to capture multi-scale temporal context.
    Input shape: (B, T, num_features)
    Output shape: (B, forecast_hours)
    """
    def __init__(
        self,
        input_size: int = 7,
        num_channels: list = [32, 64, 64, 64],
        kernel_size: int = 3,
        forecast_hours: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        in_c = input_size
        for i, out_c in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalResidualBlock(in_c, out_c, kernel_size, dilation, dropout))
            in_c = out_c
            
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Linear(32, forecast_hours)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute input to (B, C, T) for 1D convolutions
        x_perm = x.permute(0, 2, 1)
        feat = self.tcn(x_perm)
        out = self.head(feat)
        return out


class CNNExpert(nn.Module):
    """
    1D CNN Expert: Multi-kernel convolutional feature extractor for rapid local pattern 
    and demand spike detection.
    Input shape: (B, T, num_features)
    Output shape: (B, forecast_hours)
    """
    def __init__(
        self,
        input_size: int = 7,
        forecast_hours: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, forecast_hours)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute to (B, C, T)
        x_perm = x.permute(0, 2, 1)
        
        h1 = F.relu(self.bn1(self.conv1(x_perm)))
        h1 = self.pool(h1)
        
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h2 = self.pool(h2)
        
        h3 = F.relu(self.bn3(self.conv3(h2)))
        h3 = self.dropout(h3)
        
        out = self.head(h3)
        return out
