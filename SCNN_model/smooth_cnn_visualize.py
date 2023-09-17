#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:15:48 2023

@author: st_ko
"""

import glob
import keras as ks
import pandas as pd
import numpy as np
from collections import namedtuple
import tensorflow as tf
import datetime as dt
import os
from scipy import stats
import matplotlib.pyplot as plt
import glob
from utils import *
from build_multistep_model_2 import *
from sklearn.preprocessing import MinMaxScaler





# for future data (unknown)
def visualize_future_horizon(curs):


# add the list of currencies that we want to plot here
 for cur in curs:

    # finds all the predictions for all frequencies , training_lengths --> that correspond to a certain currency
    y_path = glob.glob(os.path.join('./predictions/scnn/multi_step/slide', cur ,"**/*.csv"),recursive=True)



    for p in y_path :
        freq = p.split('/')[-3]
        series_length = p.split('/')[-2]

        # get the predictions
        y_model_1 = pd.read_csv(p, header=0, index_col=0).loc[:,'F1':].transpose()


        # pick the history (-100) is just to show a lot of previous history , you can set it smaller
        history = pd.read_csv(os.path.join('./dataset',f"{freq}.csv"),
                              header=0,index_col=0).loc[:,cur][-100:].transpose()

        fig = plt.figure()
        plt.plot(history,'o-',color ='green',label='original history')
        plt.plot(y_model_1,'o-',color='purple',label = 'predictions')
        plt.title(f'FORECASTING with currency : {cur} , frequency : {freq}, training_length : {series_length}' )
        plt.xticks(rotation=90)
        plt.legend()

        # save images of visualizations of predictions
        if not (os.path.exists('./visualizations/scnn/multi_step/slide')):
                os.makedirs('./visualizations/scnn/multi_step/slide')
        plt.savefig(os.path.join('./visualizations/scnn/multi_step/slide',
                                 cur +'_' + freq + '_' + str(series_length) + '.png'))
        plt.close()




# for already existent data (plot predictions vs true data)
def visualize_past_horizon(frequencies):


    #------------------ DATA FOR EVALUATION -------------------------#
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


           # fix and align the indices
           horizon = pred_dict[freq][1]
           history = pred_dict[freq][0][cur][: horizon * 2].transpose()
           selected_index = history[:horizon].index


           # read model predictions and set the predictions index same as the data index to align the true values with the predictions
           y_model_1 = pd.read_csv(p, header=0, index_col=0).iloc[:,: horizon].transpose()
           y_model_1.index = selected_index



           fig = plt.figure()
           plt.plot(history,'o-',color ='green',label='original history')
           plt.plot(y_model_1,'o-',color='purple',label = 'predictions')
           plt.title(f'FORECASTING with currency : {cur} , frequency : {freq}, training_length : {series_length}' )
           plt.xticks(rotation=90)
           plt.legend()

           # save images
           if not (os.path.exists('./visualizations/scnn/multi_step/slide')):
                   os.makedirs('./visualizations/scnn/multi_step/slide')
           plt.savefig(os.path.join('./visualizations/scnn/multi_step/slide',
                                    cur +'_' + freq + '_' + str(series_length) + '.png'))
           plt.close()



# run as main
if __name__=="__main__":


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


    #-------------- PLOT UNKNOWN FUTURE VALUES -----------------#
    #visualize_future_horizon(frequencies['daily'][1].columns)

    #-------------- PLOT EVALUATION PLOTS, PREDICTED DATA VS ACTUAL DATA-----#
    #visualize_past_horizon(frequencies)
