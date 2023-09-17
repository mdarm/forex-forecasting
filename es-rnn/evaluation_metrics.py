import numpy as np


def mse(y, y_hat):
    """
    Calculates Mean Squared Error.

    Parameters:
    - y (numpy array): Actual test values.
    - y_hat (numpy array): Predicted values.

    Returns:
    - float: Mean squared error.
    """
    y     = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    mse   = np.mean(np.square(y - y_hat)).item()
    return mse

def smape(y, y_hat):
    """
    Calculates Symmetric Mean Absolute Percentage Error.

    Parameters:
    - y (numpy array): Actual test values.
    - y_hat (numpy array): Predicted values.

    Returns:
    - float: Symmetric mean absolute percentage error.
    """
    y     = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    smape = np.mean(2.0 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))
    return smape * 100
