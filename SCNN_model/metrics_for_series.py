#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 21:56:11 2023

@author: st_ko
"""



import keras as ks
import pandas as pd
import numpy as np
from collections import namedtuple
import tensorflow as tf
import datetime as dt
import os
from scipy import stats
from utils import *
from sklearn.preprocessing import MinMaxScaler
from build_multistep_model_2 import *
import glob




# simple smape metric
def smape(history, predictions):
    return 1/len(history) * np.sum(2 * np.abs(history-predictions) / (np.abs(history) + np.abs(predictions))*100)



# simple mse
def mse(y, y_hat):
    y  = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    mse   = np.mean(np.square(y - y_hat)).item()
    return mse




# this is a function to make predictions on history data so we can then evaluate them
def predict_on_history(frequencies):
    # TODO : NOW i have to
    # 1) make predictions on horizons chosen from the train dataset , for each currency series ,each frequency and for each series_length
    # 2) begin by making predictions for the last horizon values in the dataset
    # 3) then calculate MASE , SMAPE for all the horizon using all series_lengths
    # 4) ideally we do not want to use a set that has been used in training
    # so i will take the previous years ...-2009 and use the history to predict the next horizons
    # test for already seen data so we could evaluate the models
    #--------------------- data for evaluation ---------------------------#
    dseries = pd.DataFrame(frequencies['daily'][1].loc[:'2009-12-31'])
    wseries = pd.DataFrame(frequencies['weekly'][1].loc[:'2009-12-27'])
    mseries = pd.DataFrame(frequencies['monthly'][1].loc[:'2009-12-31'])
    qseries = pd.DataFrame(frequencies['quarterly'][1].loc[:'2009-12-31'])
    yseries = pd.DataFrame(frequencies['yearly'][1].loc[:'2009-12-31'])
    #---------------------------------------------------------------------#


    pred_dict = {'daily' : dseries,
                 'weekly' : wseries,
                 'monthly' : mseries,
                 'quarterly' : qseries,
                 'yearly' : yseries}

    # define the currencies
    series = frequencies['daily'][1]
    currencies = series.columns


    # dictionary of predictions
    prediction = { x : {} for x in currencies}


    #we need to read all the models
    models = sCnn()

    for cur in prediction.keys():


            for m in models:


                # we will pick the dataset that we have not observed so , 2008 - 2010
                series = pd.DataFrame(pred_dict[m.freq_name][cur])


                # pick index frequency , business frequency
                f_s = frequencyCalc(m.freq_name)
                series.index = pd.DatetimeIndex(series.index)
                series = series.to_period(f_s)


                series_n = np.reshape(series,(-1,1))




                minmax = MinMaxScaler().fit(series_n)
                series_norm = minmax.transform(series_n)



                series_norm = pd.DataFrame(series_norm, index=series.index, columns = series.columns)


                # find the optimum alpha for this currency and apply the exponential smoothing
                optimum_a = optimum_al(series_norm)
                _,smoothed_series = exponential_smooth(series_norm,optimum_a,m.freq_name,Hw=False)


                prediction[cur][m.freq_name] = {}


                print(f"\n------------------PREDICTING CURRENCY : {cur} --------------------\n")

                # for all training lengths
                for series_length in m.training_lengths:


                     # pick the history we need to use to predict the next horizon
                     x_train = smoothed_series.iloc[-series_length:,:]


                     # initialize dict
                     prediction[cur][m.freq_name][series_length] = np.zeros([1,m.horizon])

                     # pointer
                     curr_prediction = prediction[cur][m.freq_name][series_length]


                     # ---------- TF KERAS SET UP ------------------------------------------------------#
                     ks.backend.clear_session()
                     tf.compat.v1.reset_default_graph()
                     # load the particular model
                     model_file = os.path.join(f'./trained_models/scnn/multi_step/slide/{cur}/{m.freq_name}/{series_length}',
                                               '{}_length_{}.h5'.format(m.freq_name, series_length))
                     est = ks.models.load_model(model_file)
                     # ---------------------------------------------------------------------------------#

                     # turn x_train into array to feed to model
                     x_train = np.array(x_train)
                     x_train = np.reshape(x_train,(1,-1))

                     # fill prediction
                     curr_prediction[:,:m.horizon] = est.predict(x_train)



                     # we want to fit the inverse transform here to destandardize the data
                     prediction_df  =  pd.DataFrame(curr_prediction.copy())
                     prediction_denorm = np.array(prediction_df)
                     prediction_denorm = np.reshape(prediction_denorm,(-1,1))
                     # denormalize
                     prediction_denorm = minmax.inverse_transform(prediction_denorm)
                     final_prediction = np.reshape(prediction_denorm,(1,-1))
                     prediction[cur][m.freq_name][series_length]= final_prediction




                     #------------ save prediction csvs for all models ------------------#
                     output = pd.DataFrame(index=[0], columns=['F' + str(i) for i in range(1, 49)])
                     output.index.name = 'id'

                     # fill dataframe with prediction
                     output.iloc[:, : m.horizon] = prediction[cur][m.freq_name][series_length]

                     # save predictions
                     if not (os.path.exists(os.path.join('./predictions','scnn/multi_step/slide',cur,m.freq_name,str(series_length )))):
                       os.makedirs(os.path.join('./predictions','scnn/multi_step/slide',cur,m.freq_name,str(series_length) ))
                     output.to_csv(os.path.join('./predictions','scnn/multi_step/slide',cur,m.freq_name,str(series_length) , 'prediction.csv'))




# function to evaluate the predictions
# it reads the predictions , finds the corresponding true values and calculates the 2 metrics
# then it saves csv files for the different currencies/frequencies/training lengths

def evaluate(frequencies):

    # create dataframe for results
    metrics = np.zeros((1,2))

    #------------------ DATA FOR EVALUATION -------------------------#
    # we want data we have not used so we go backwards
    dseries = pd.DataFrame(frequencies['daily'][1].loc['2009-12-31':])
    wseries = pd.DataFrame(frequencies['weekly'][1].loc['2009-12-27':])
    mseries = pd.DataFrame(frequencies['monthly'][1].loc['2009-12-31':])
    qseries = pd.DataFrame(frequencies['quarterly'][1].loc['2009-12-31':])
    yseries = pd.DataFrame(frequencies['yearly'][1].loc['2009-12-31':])
    #------------------------------------------------------------------#

    pred_dict = {'daily' : (dseries, 14),
                 'weekly' : (wseries,13),
                 'monthly' : (mseries,18),
                 'quarterly' : (qseries,8),
                 'yearly' : (yseries,6) }


    # define the currencies
    series = frequencies['daily'][1]
    curs = series.columns


    for cur in curs:

       y_path = glob.glob(os.path.join('./predictions/scnn/multi_step/slide', cur ,"**/*.csv"),recursive=True)


       for p in y_path :
           #get the frequenxy so i know which dataset to plot with
           freq = p.split('/')[-3]
           series_length = p.split('/')[-2]


           # select y_trye and y_predicted
           horizon = pred_dict[freq][1]
           history = pred_dict[freq][0][cur][: horizon ].transpose()
           #selected_index = history[:horizon].index


           # read model predictions and set the predictions index same as the data index to align the true values with the predictions
           y_model_1 = pd.read_csv(p, header=0, index_col=0).iloc[:,: horizon].transpose()


           # transform into arrays to calculate the metrics
           y_model_1 = np.array(y_model_1)
           history = np.array(history)


           smape_result = smape(history,y_model_1)
           mse_result = mse(history,y_model_1)

           # create dataframe of results
           metrics = pd.DataFrame ( {"smape":smape_result,
                                     "mse":mse_result},index = [0])


           # save csv of metric to cur-> frequency -> series_length directory
           if not (os.path.exists(os.path.join('./evaluation','scnn/multi_step/slide',cur,freq,str(series_length )))):
             os.makedirs(os.path.join('./evaluation','scnn/multi_step/slide',cur,freq,str(series_length) ))
           metrics.to_csv(os.path.join('./evaluation','scnn/multi_step/slide',cur,freq,str(series_length) , 'evaluation.csv'))







if __name__=="__main__":

    # Let's now read all the different frequency - datasets
    # for each frequency we take the end of the month
    frequencies = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q',
        'yearly': 'Y'
    }

    for freq_name, freq_code in frequencies.items():
        data = pd.read_csv(f"./dataset/{freq_name}.csv",index_col='Date')
        frequencies[freq_name] = (frequencies[freq_name],data)



    # 1)predict 2) evaluate , run them with this order
    predict_on_history(frequencies)
    evaluate(frequencies)
