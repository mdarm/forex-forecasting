Vector autoregression (V-AR) is a statistical model used to capture the relationship between multiple quantities as they change over time. 
It is a type of stochastic process model that generalizes for the single-variable (univariate) autoregressive model by allowing for multivariate time series.

It should be noted that all variables have to be of the same order of integration. The following cases are distinct:

- All variables are I(0) (stationary): this is in the standard case, i.e. a VAR in level
- All variables are I(d) (non-stationary) with d > 0.
  - The variables are cointegrated: the error correction term has to be included in the VAR.
  - The variables are not cointegrated: first, the variables have to be differenced d times and one has a VAR in difference.
 
As part of its parameter optimization, an augmented Dickey-Fuller test is performed, in order to decide the order of differences, as well as multiple AIC tests, in order to find the optimal lag order.

In general, our V-AR model achieves a sMAPE (symmetric Mean Absolute Percentage Error) of around 0.5-3%, depending on the:
- currency,
- seasonality,
- training set size.


Expanding to V-ARMA could potentially further improve performance. The model serves as a benchmark for the various NN models' performance.
