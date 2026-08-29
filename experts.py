import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMExpert(nn.Module):
    """LSTM expert to capture long-term temporal dependencies."""
    def __init__(self, input_dim, hidden_dim, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        out, (hn, cn) = self.lstm(x)
        # Take the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class Chomp1d(nn.Module):
    """Helper module to ensure causal convolutions in TCN."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNExpert(nn.Module):
    """TCN expert for efficient temporal modeling."""
    def __init__(self, input_dim, num_channels, kernel_size, pred_len):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                          stride=1, dilation=dilation_size, padding=padding),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(0.2) # Keeping internal dropout
            ]
        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_channels[-1], pred_len)

    def forward(self, x):
        # Transpose for Conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2) 
        out = self.network(x)
        # Global pooling (take the last step feature)
        out = out[:, :, -1]
        out = self.dropout(out)
        out = self.fc(out)
        return out

class CNNExpert(nn.Module):
    """CNN expert to extract local/spatial features from load windows."""
    def __init__(self, input_dim, hidden_dim, pred_len):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, pred_len)

    def forward(self, x):
        # Transpose for Conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        # Adaptive pool down to 1 feature per channel
        out = self.pool(out).squeeze(-1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
