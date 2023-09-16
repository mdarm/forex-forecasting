import numpy as np


def smape(y, y_hat):
    """
    Calculates Symmetric Mean Absolute Percentage Error.

    Parameters:
    - y (numpy array): Actual test values.
    - y_hat (numpy array): Predicted values.

    Returns:
    - float: Symmetric mean absolute percentage error.
    """
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    smape = np.mean(2.0 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))

    return smape


def mase(y, y_hat, y_train, seasonality=5):
    """
    Calculates Mean Absolute Scaled Error.

    Parameters:
    - y (numpy array): Actual test values.
    - y_hat (numpy array): Predicted values.
    - y_train (numpy array): Actual train values for Naive1 predictions.
    - seasonality (int): Main frequency of the time series. 
            Quarterly 4, Daily 7 (5 for forex; no weekends), Monthly 12.

    Returns:
    - float: Mean absolute scaled error.
    """
    y_hat_naive = []
    for i in range(seasonality, len(y_train)):
        y_hat_naive.append(y_train[(i - seasonality)])

    masep = np.mean(abs(y_train[seasonality:] - y_hat_naive))
    mase = np.mean(abs(y - y_hat)) / masep
    return mase
