import numpy as np


def mse(y, y_hat):
    """
    Calculates Mean Squared Error.

    Parameters:
    - y (numpy array): Actual test values.
    - y_hat (numpy array): Predicted values.

    Returns:
    - float: Mean Squared Error.
    """
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    mse = np.mean(np.square(y - y_hat)).item()
    return mse


def mape(y, y_hat):
    """
    Calculates Mean Absolute Percentage Error.

    Parameters:
    - y (numpy array): Actual test values.
    - y_hat (numpy array): Predicted values.

    Returns:
    - float: Mean Absolute Percentage Error.
    """
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    mape = np.mean(np.abs(y - y_hat) / np.abs(y))
    return mape


def smape(y, y_hat):
    """
    Calculates Symmetric Mean Absolute Percentage Error.

    Parameters:
    - y (numpy array): Actual test values.
    - y_hat (numpy array): Predicted values.

    Returns:
    - float: Symmetric Mean Absolute Percentage Error.
    """
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    smape = np.mean(2.0 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))
    return smape


def mase(y, y_hat, y_train, seasonality):
    """
    Calculates Mean Absolute Scaled Error.

    Parameters:
    - y (numpy array): Actual test values.
    - y_hat (numpy array): Predicted values.
    - y_train (numpy array): Actual train values for Naive1 predictions.
    - seasonality (int): Main frequency of the time series (Quarterly 4, Daily 7, Monthly 12).

    Returns:
    - float: Mean Absolute Scaled Error.
    """
    y_hat_naive = []
    for i in range(seasonality, len(y_train)):
        y_hat_naive.append(y_train[(i - seasonality)])

    masep = np.mean(abs(y_train[seasonality:] - y_hat_naive))
    mase = np.mean(abs(y - y_hat)) / masep
    return mase
