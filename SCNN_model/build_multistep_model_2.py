#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 02:48:36 2023

@author: st_ko
"""



import keras as ks
import pandas as pd
import numpy as np
#import metrics
from collections import namedtuple
import tensorflow as tf
import datetime as dt
import os
import tensorflow as tf






# new convolutional daily model
# TODO : 1) test different lucas numbers for the number GN of hidden layers (from the paper : 1,3,4,7,11,18,29,47,76,123...)
# TODO : 2) strides in conv, maxpool, dense layers are unknown from the paper, we used 1
# TODO : 3) dropout is not known exactly where we apply it . We applied it after every layer ro reduce overfitting


def scnn_model(series_length,bs,horizon,epochs=1000, GN=11):

    input = ks.layers.Input((series_length,1))

    conv1 = tf.keras.layers.Conv1D(filters = 128 ,kernel_size=2,strides = 1, padding = 'valid',activation ='relu')(input)

    # add dropout
    conv1 = tf.keras.layers.Dropout(0.2)(conv1,training=True)

    conv2 = tf.keras.layers.Conv1D(filters = 128 ,kernel_size=2,strides = 1, padding = 'valid',activation = 'relu')(conv1)


    # add dropout
    conv2 = tf.keras.layers.Dropout(0.2)(conv2,training=True)


    conv3 = tf.keras.layers.Conv1D(filters = 128 ,kernel_size=2,strides = 1, padding = 'valid',activation = 'relu')(conv2)

    #add dropout
    conv3 = tf.keras.layers.Dropout(0.2)(conv3,training=True)


    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid")(conv3)
    flatten1 = ks.layers.Flatten()(pool1)


    # generate a sequence of  N hidden_layers based on number N chosen from golden ratio (retrieved from paper)
    # we will give the N number outside the model
    # add the flattened layer in the list --> we want recursively each next dense to be applied to each previous one
    # intantiate the layers
    hidden_layers =[flatten1]
    for l in range(1,GN + 1)  :
        hl = ks.layers.Dense(100,activation='relu')
        hidden_layers.append(hl)

    # apply the layers
    for i in range(1,GN + 1):
        hl = hidden_layers[i](hidden_layers[i-1])
        # add dropout
        hl = tf.keras.layers.Dropout(0.2)(hl,training=True)
        # save output
        hidden_layers[i] = hl

    # take the last layer for horizon prediction
    output = ks.layers.Dense(horizon,activation='relu')(hidden_layers[-1])

    est = ks.Model(inputs=input, outputs=output,name='sCnn')
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    epochs = epochs
    batch_size =  bs
    return est, epochs, batch_size





# build the models here , named tuple of characteristic
def sCnn():
    Model = namedtuple('Model', ['freq_name', 'horizon', 'model_constructor', 'training_lengths'])

    daily = Model('daily', 14, scnn_model, [14,20,240]  )
    weekly = Model('weekly', 13, scnn_model, [52,13,26]  )

    monthly = Model('monthly', 18, scnn_model, [24,18] )
    quarterly = Model('quarterly', 8, scnn_model, [12,8]  )
    yearly = Model('yearly', 6, scnn_model, [6] )

    return [quarterly,yearly,monthly,weekly,daily]


# for testing
if __name__ == "__main__":
    model1 = sCnn()
    m = model1[0].model_constructor
    m1,ep,bs = m(32,1,14)
    print(m1.summary())
