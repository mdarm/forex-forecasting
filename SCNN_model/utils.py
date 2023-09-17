
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import os
import keras as ks
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from IPython.display import clear_output
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing , HoltWintersResults






# function to implement the optimum "a" for exponential smoothing
# as described in the paper
def optimum_al(series):
    opt_a =(series.max() - series.min() - series.mean() ) / (series.max() - series.min())
    return opt_a

# function to pick the seasonal periods based on the frequency dataset selected
# this is in case someone uses the holt winters exponential smoothing method (capture seasonality,level,..)
def seasonals(freq_name):
    if(freq_name == "daily"):
        return 5
    elif(freq_name == "weekly"):
        return 4
    elif(freq_name == "monthly"):
        return 12
    elif(freq_name == 'quarterly'):
        return 4
    else :
        return 1


# exponential smooth with the optimum alpha calculated from the suggested
# formula , also it offers the option for wolt winters smoothing
# we tested it , it did not improve the results
def exponential_smooth(series,optimum_a,freq_name,Hw=False):

    temp = np.zeros((series.shape[0],series.shape[1]))

    for i,c in enumerate(series.columns):


           if(Hw==False):
                #--------------------------- simple smoothing --------------------------- #
                sm = SimpleExpSmoothing(series[c], initialization_method="estimated").fit(
                              smoothing_level=optimum_a[c], optimized=False)
                #-------------------------------------------------------------------------#
                temp[:,i] = sm.fittedvalues
                mod = sm
           else :
                hw = ExponentialSmoothing(
                    series[c], trend="add", seasonal="add"
                    , initialization_method='estimated',
                      seasonal_periods = seasonals(freq_name)
                    ).fit(optimized=True)

                temp[:,i] = hw.fittedvalues
                mod = hw
            # transform to dataframe again and return it
    smoothed_series = pd.DataFrame(temp,index=series.index , columns = series.columns)
    return mod,smoothed_series


# function to calculate frequency for dataframes
def frequencyCalc(freq_name):
    if(freq_name == "daily"):
        return 'B'
    elif(freq_name == "weekly"):
        return 'W'
    elif(freq_name == "monthly"):

        return 'M'
    elif(freq_name == 'quarterly'):

        return 'Q'
    else :

        return 'A'


if __name__ =="__main__":
    pass
