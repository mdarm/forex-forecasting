import torch
import torch.nn.functional as F


def sMAPE(y_pred, y_true):
    """
    Compute Symmetric Mean Absolute Percentage Error (sMAPE).
    """
    num = torch.abs(y_pred - y_true)
    denom = (torch.abs(y_pred) + torch.abs(y_true)) / 2
    smape_val = num / (denom + 1e-10)  # added epsilon to avoid division by zero
    return 2 * torch.mean(smape_val)

def MASE(training_series, y_pred, y_true):
    """
    Compute Mean Absolute Scaled Error (MASE).
    """
    n = training_series.shape[0]
    d = torch.abs(y_true - y_pred)
    scale = torch.mean(torch.abs(training_series[1:n] - training_series[0:n-1])) + 1e-10  # added epsilon to avoid division by zero
    mase_val = d / scale
    return torch.mean(mase_val)

def hybrid_loss(training_series, y_pred, y_true):
    """
    Hybrid loss function combining sMAPE and MASE.
    """
    return 0.5 * sMAPE(y_pred, y_true) + 0.5 * MASE(training_series, y_pred, y_true)
