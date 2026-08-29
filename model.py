import torch
import torch.nn as nn
from experts import LSTMExpert, TCNExpert, CNNExpert
from context import ContextEncoder, GatingNetwork

class CAEGNet(nn.Module):
    """
    Context-Adaptive Expert Gating Network (CAEG-Net) 
    with Closed-Loop Error Feedback.
    """
    def __init__(self, input_dim=1, seq_len=168, pred_len=24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Base Experts
        self.expert_lstm = LSTMExpert(input_dim=input_dim, hidden_dim=64, num_layers=2, pred_len=pred_len)
        self.expert_tcn = TCNExpert(input_dim=input_dim, num_channels=[32, 64, 64], kernel_size=3, pred_len=pred_len)
        self.expert_cnn = CNNExpert(input_dim=input_dim, hidden_dim=64, pred_len=pred_len)
        
        # 2. Context Extraction & Gating
        self.context_encoder = ContextEncoder(seq_len=seq_len, pred_len=pred_len)
        self.gating_network = GatingNetwork(context_dim=16, num_experts=3)
        self.error_proj = nn.Linear(pred_len, pred_len)
        
    def forward(self, x_load, prev_error):
        """
        x_load: [batch_size, seq_len, input_dim] - Historical load
        prev_error: [batch_size, pred_len] - Previous prediction errors
        """
        # Step 1: Expert Predictions
        out_lstm = self.expert_lstm(x_load) # [batch_size, pred_len]
        out_tcn = self.expert_tcn(x_load)   # [batch_size, pred_len]
        out_cnn = self.expert_cnn(x_load)   # [batch_size, pred_len]
        
        # Stack predictions: [batch_size, 3, pred_len]
        expert_outputs = torch.stack([out_lstm, out_tcn, out_cnn], dim=1)
        
        # Step 2: Context Encoding
        context_emb = self.context_encoder(x_load, prev_error)
        
        # Step 3: Dynamic Gating Weights
        weights = self.gating_network(context_emb) # [batch_size, 3]
        
        # Step 4: Weighted Fusion
        # Reshape weights for broadcasting: [batch_size, 3, 1]
        weights_expanded = weights.unsqueeze(-1)
        
        # Final prediction: Weighted sum over experts
        final_out = torch.sum(weights_expanded * expert_outputs, dim=1) # [batch_size, pred_len]
        
        # Closed-loop residual error correction
        # Subtract the previous error to pull the prediction back toward the target
        final_out = final_out - 0.85 * prev_error
        
        # Return final_out and weights (useful for interpretability/logging)
        return final_out, weights
