import torch
import torch.nn.functional as F


def smape(y_pred, y_true):
    """
    Compute Symmetric Mean Absolute Percentage Error (sMAPE).

    Parameters:
    - y_pred (torch.Tensor): The predicted values.
    - y_true (torch.Tensor): The true values.

    Returns:
    - torch.Tensor: The sMAPE loss value.
    """
    num = torch.abs(y_pred - y_true)
    denom = (torch.abs(y_pred) + torch.abs(y_true)) / 2
    smape_val = num / (denom + 1e-10)

    return 2 * torch.mean(smape_val)


def mase(training_series, y_pred, y_true):
    """
    Compute Mean Absolute Scaled Error (MASE).

    Parameters:
    - training_series (torch.Tensor): The training time series data.
    - y_pred (torch.Tensor): The predicted values.
    - y_true (torch.Tensor): The true values.

    Returns:
    - torch.Tensor: The MASE loss value.
    """
    n = training_series.shape[0]
    d = torch.abs(y_true - y_pred)

    scale = torch.mean(torch.abs(training_series[1:n] - training_series[0:n-1])) + 1e-10
    mase_val = d / scale

    return torch.mean(mase_val)


def hybrid_loss(training_series, y_pred, y_true):
    """
    Hybrid loss function combining sMAPE and MASE.

    Parameters:
    - training_series (torch.Tensor): The training time series data.
    - y_pred (torch.Tensor): The predicted values.
    - y_true (torch.Tensor): The true values.

    Returns:
    - torch.Tensor: The hybrid loss value.
    """
    return 0.5 * smape(y_pred, y_true) + 0.5 * mase(training_series, y_pred, y_true)


def compute_alpha(training_series):
    """
    Compute the alpha scaling factor using the simple naive method.
    
    Parameters:
    - training_series (torch.Tensor): The training time series data.
    
    Returns:
    - torch.Tensor: The computed scaling factor for the MSIS loss.
    """
    naive_forecast = torch.roll(training_series, shifts=1)
    errors = training_series[1:] - naive_forecast[1:]
    
    alpha = torch.mean(torch.abs(errors))
    
    return alpha


def msis(L, U, y_true, alpha):
    """
    Compute the Mean Scaled Interval Score (MSIS) loss.
    
    Parameters:
    - L (torch.Tensor): Lower bound predictions.
    - U (torch.Tensor): Upper bound predictions.
    - y_true (torch.Tensor): Actual values.
    - alpha (torch.Tensor): Scaling factor derived from in-sample one-step-ahead forecast errors.
    
    Returns:
    - torch.Tensor: The MSIS loss value.
    """
    term1 = (U - L) / alpha
    term2 = 2.0 / alpha * (L - y_true) * (y_true < L).float()
    term3 = 2.0 / alpha * (y_true - U) * (y_true > U).float()

    loss = term1 + term2 + term3
    
    return loss.sum()
