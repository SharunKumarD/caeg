import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    """
    Extracts four explicit features from the input window:
    1. Trend strength
    2. Volatility level
    3. Periodicity strength
    4. Recent prediction error (closed-loop feedback)
    """
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Project the 4 concatenated scalar features into a dense representation
        self.proj = nn.Linear(4, 16)
        
    def forward(self, x_load, prev_error):
        # x_load: [batch_size, seq_len, 1]
        # prev_error: [batch_size, pred_len]
        
        batch_size = x_load.size(0)
        
        # 1. Trend strength: Simple difference between end and start of window
        trend = (x_load[:, -1, 0] - x_load[:, 0, 0]).unsqueeze(-1) # [batch_size, 1]
        
        # 2. Volatility level: Standard deviation of the sequence
        volatility = x_load[:, :, 0].std(dim=1).unsqueeze(-1) # [batch_size, 1]
        
        # 3. Periodicity strength: Autocorrelation proxy (e.g., lag 24 for daily periodicity)
        # Using cosine similarity between two segments separated by lag
        lag = min(24, self.seq_len // 2)
        if lag > 0:
            part1 = x_load[:, lag:, 0]
            part2 = x_load[:, :-lag, 0]
            # Add eps to avoid NaN on zero vectors
            periodicity = torch.cosine_similarity(part1, part2, dim=1, eps=1e-8).unsqueeze(-1)
        else:
            periodicity = torch.zeros((batch_size, 1), device=x_load.device)
            
        # 4. Recent prediction error (closed-loop feedback):
        # Average absolute error from the previous prediction window
        recent_error = prev_error.abs().mean(dim=1, keepdim=True) # [batch_size, 1]
        recent_error = torch.tanh(recent_error.view(-1, 1))
        
        # Concatenate features
        context_features = torch.cat([trend, volatility, periodicity, recent_error], dim=1) # [batch_size, 4]
        
        # Encode into dense representation
        context_emb = torch.relu(self.proj(context_features)) # [batch_size, 16]
        
        return context_emb

class GatingNetwork(nn.Module):
    """
    Lightweight MLP designed from scratch to output dynamic expert weights 
    based on the extracted context.
    """
    def __init__(self, context_dim=16, num_experts=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_experts)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, context_emb):
        # context_emb: [batch_size, context_dim]
        logits = self.mlp(context_emb)
        weights = self.softmax(logits) # [batch_size, num_experts]
        return weights
