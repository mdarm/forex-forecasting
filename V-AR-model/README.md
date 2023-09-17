Vector autoregression (VAR) is a statistical model used to capture the relationship between multiple quantities as they change over time. 
It is a type of stochastic process model that generalizes for the single-variable (univariate) autoregressive model by allowing for multivariate time series.

It should be noted that all variables have to be of the same order of integration. The following cases are distinct:

- All variables are I(0) (stationary): this is in the standard case, i.e. a VAR in level
- All variables are I(d) (non-stationary) with d > 0.
  - The variables are cointegrated: the error correction term has to be included in the VAR.
  - The variables are not cointegrated: first, the variables have to be differenced d times and one has a VAR in difference.
