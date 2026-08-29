import torch

def MAE(pred, target):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target))

def RMSE(pred, target):
    """Root Mean Square Error"""
    return torch.sqrt(torch.mean((pred - target) ** 2))

def R2(pred, target):
    """R-squared (Coefficient of Determination)"""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2

def MSE(pred, target):
    """Mean Squared Error"""
    return torch.mean((pred - target) ** 2)

def evaluate_all(pred, target):
    """
    Compute and return all metrics as a dictionary.
    pred: [batch_size, pred_len]
    target: [batch_size, pred_len]
    """
    return {
        'MSE': MSE(pred, target).item(),
        'MAE': MAE(pred, target).item(),
        'RMSE': RMSE(pred, target).item(),
        'R2': R2(pred, target).item()
    }
