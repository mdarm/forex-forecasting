<a name="readme-top"></a>

# Smoothed Convolutional Neural Models for foreign exchange Time series forecasting

## About the Project
This project is inspired from the following publication:
(https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00599-y)
It is an attempt to design and implement an exponential smoothing , convolutional
network pipeline that receives a time series , applies exponential smoothing and
feeds it to a convolutional network to do the final projections.The model is being
tested on the ECB  foreign exchange dataset which includes the exchange rates of many
currencies.The task we have is to try to forecast the exchange rate of the currencies.

## Prerequisites
The models were tested with :
```
python==3.8.10
tensorboard==2.13.0
tensorboard-data-server==0.7.1
tensorflow==2.13.0
tensorflow-estimator==2.13.0
tensorflow-io-gcs-filesystem==0.32.0
scikit-learn==1.2.2
scipy==1.10.1
sklearn==0.0.post1
pandas==2.0.3
numpy==1.24.2
matplotlib==3.7.1
matplotlib-inline==0.1.6
```
In a clean python virtual environment you can just run `pip install <The above list in a txt file>` to install them altogether.

Regarding Tensorflow in particular the models were implemented using V1 compatibility mode.

## Built With

The code and models were built using :
* python
* Keras
* Tensorflow

## Usage
1. To download the data you can run the `download_and_resample.py` which downloads the latest version of the dataset
and moves it into the path you are located in ,`process_data.py` and `fetch_data.py` must also be in the same directory with `download_and_resample.py`.
2. Alternatively we provide already split frequency datasets and keep in mind that the training and validation
is done on the old dataset , so it will be easier to download the frequency datasets the from links(https://drive.google.com/drive/folders/1iRNA97ZZDvIN8A66dgbFP2q886GMu5kR?usp=sharing).Either way in order to get it working you need to put the frequency split datasets (weekly.csv,daily.csv...) into a folder called `dataset` in the same location as the `.py` files.
3. To train the dataset you run `train_smooth_cnn.py` which will train the models and save them as `.h5` into a folder called `trained_models`.
`trained_models` and all other directories will be created into the same directory the .py files are in.
4. After training is done you can make `predict_smooth_cnn_2.py` to make future future predictions and save them for each curreny, for each frequency,and for each different training_length(history) into a directory called `predictions`.
5. If you want to make predictions on past data and validate the model you run `metrics_for_series.py` . This will pick a fixed split of the dataset and use the trained models to predict . Then it will calculate the mse and smape between the true values and the predictions and save them into a directory called `evaluation` , for each currency,frequency and training_length.
6. If at any time you have made predictions and want to visualize them you have 2 options . If you have predicted past events and you want
to plot both the true values and the predictions as well you must go in the script `smooth_cnn_visualize.py` and comment out the line
`visualize_future_horizon(frequencies['daily'][1].columns)` while uncommenting `visualize_past_horizon(frequencies)`.If you have predicted
future values where the values are not known you do the opposite. The visualizations will be made in a directory called `visualizations`.
7. In (https://drive.google.com/drive/folders/1iRNA97ZZDvIN8A66dgbFP2q886GMu5kR?usp=sharing) you will also find the visualizations of the model training losses with `hidden_layers = 76` and the future predictions of this model.Additionally you will find a set of evaluation results of the model on past observations with mse and smape as the metrics, one `.csv` in each directory for each frequency , training_length and currency.Provided is also a set of some visualizations on past observations using this models for some currencies. I also provide the 'USD'
series prediction models for quarterly frequency .
8. `data_analysis_3c.py` is a script that was only used to plot the series in different formats , but i include it for completeness.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Introduction
Cnn usually extract features from pixels although in Time series altough in Time series, additional features are not always available. Also due to weight sharing, cnn models reduce the trainable parameters in comparison to simple MLP models as well as LSTM,GRU,RNN.Cnn also offer the ability to design dilated convolutions in order to find the relations between not always the closest points.Many Experiments have utilized Convolutional neural networks in various combinations for forecasting tasks, like
Autoregressive Convolutional Neural Networks for Asynchronous Time Series where convolutional neural networks were combined with an autoregressive model , ourperforming both CNN and LSTM."Wavenet" is also a popular architecture which outperformed classic timeseries methods and LSTM as well.A framework was established to combine CNN-LSTM with simple exponential smoothing.Studies have shown that combinations of Smoothing the Time Series and then feeding it to Convolutional Neural Networks can lead to quality forecasts.
In this work we draw from the aforementioned publication and implement a pipeline of Simple exponential smoothing the Time Series , then feeding them to the Cnn and receving the final projections.
The rest of this work is organized as follows.First we present the Exponential Smoothing method , then we talk about the dataset and the preprocessing steps. After that we move on to the Cnn architecture designed and implemented and then we analyse and comment on the results .

## Exponential smoothing
Smoothing a Time Series in essence expresses a forecast as a weighted average of past values.The Algorithm of exponential smoothing was first suggested by Robert Goodell Brown in 1956 and then expanded by Charles C.Holt in 1957.Simple exponential smoothing can improve predictions by averaging past values .The importance of the observations decreases exponentially as one moves backwards towards the “older” historic data. The method is useful for irregular and flunctuating data.

### Implementation of the Simple Exponential Smoothing
To apply simple exponential smoothing we used the statsmodels library, where many forecasting and time series analysis methods are implemented.We instantiate a model of the class and then we call fit() into our model.The fitted values are the predicted values.Most important we have the parameter alpha also to pick and different solving options like the way we estimate the coefficients.The choice of "alpha" is based on the paper "Time-series analysis with smoothed Convolutional Neural Networks" (https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00599-y)

## Design of the Model
First we must mention the fact that the model we created is based on one dimensional series and the way we used the CNN is not in conjunction with many different time series. Instead we take one time series each time and train this specific model for this time series. This someone might argue is not very akin to convolutional neural networks’ training but the reasoning behind it is that we have different time series with potentially different characteristics and since our task is not series classification but series forecasting we believe that is is more logical to construct convolutional networks , where each one focuses on specific time series , learning the features of it.Also important the model we created is a multi step, which means it is trained to predict at once t time steps and not only one. More precisely it is trained to predict the chosen horizon, a number of steps forward into the future.

## Architecture
The mode accepts one-dimensional series of size series_length , where series_length is defined apriori and it is fixed as many methods in the M4 forecasting competition also do(https://www.sciencedirect.com/science/article/pii/S0169207019301128).The values that we choose for series_length depends on the series at hand as we will discuss in the experiments. Upon receiving the input we slide a convolutional kernel with 128 channels , kernel size of 2 and strides = 1, and we use valid padding since we don’t want to potentially add padding. The non linear activation function chosen for all the layers is the relu . We do not use sigmoid or softmax since this is not a classification task but rather a regression one. After the convolutional layer we apply dropout of 0.2 to prevent overfitting.Generally after each convolutional layer we apply dropout.After the first convolutional layer and dropout we stack the covolutional-dropout  sequence 2 more times and then we apply max pooling with pool size of 2. Next we flatten the output and we feed it into the hidden layers . Regarding the hidden layers, since there is no easy answer on how many we should use we decided to adopt the lucas numbers sequence as a list of potential hidden layers numbers.Successive Lucas numbers converge to the golden ratio and also appear in many different applications and are also used in forecasting. We can calculate them if instead of 0+1 we start the fibinacci sequence with 1 + 2. and construct the additions just like in the fibonacci sequence. In this case due to time constraints we selected the number of hidden layers from {3,11,47,76}  and trained a different model for each choice.After the hidden layers we apply a final dense layer with output shape same as the forecast horizon to get the forecasts we want.

## Dataset and Training
Regarding the Dataset and the time series available for this method , we used the Dataset from the ECB https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip?ff78e8378c020afda6dd41f057ea4246
which has all the exchange rates from 1999 until 2023 for all currencies.

1. First we split the dataset into series of different frequencies and created the daily,weekly,monthly,quarterly,yearly time series .Each new dataset contains the history of the observations for all currencies for it’s corresponding frequency.For example the yearly will contain 25 observations , since we have 25 years between 1999 and 2023, including 1999.
2. For each time series of different frequency and also of different currency, we normalize it as described and then apply the exponential smoothing.
3. We feed the output of one series into the convolutional model and train the model to predict a horizon of K steps  for this frequency, where K is a model’s designers choice .Drawing from the M4 forecasting competition guidelines we chose as forecast windows their siggested windows for each frequency.More info can be found on the paper mentioned.
4. Another question is the series_length , which means how many observations in the past the model takes into consideration during training in order to predict the future.This will be fixed for each model as mentioned before .
Since picking a very small number for series_length means we are training the model only on the most recent data, this has the  danger of feeding the model with data that are not enough for it to retrieve any potential trend or seasonality.On the other hand if the history is too large , there exists the danger that the model does not attend to the latest events adequately or it constructs periodic patterns that are not always there.In order to attempt to balance this we have used a list of series_lengths for each frequency
5. Trying to collect latest data , we picked a date ,[2010-01-04] and from the data after this index we create a new dataset for each frequency. From this dataset we construct the training and validation set, which we utilized in a generator class to yield batches of data during training.
6. For the number of lucas hidden layers we experimented with 3 , 11 , 47 and 76 and trained the model for 200 epochs for each frequency , for each training length, for each currency.

## Predictions and Results
Regarding the results of the combination of the exponential smoothing with the convolutional neural network the results were not impressive , although some long term patters were captured with the cnn , mainly for relatively stable currencies and for big frequencies like quarterly and monthlY.The results can be found on the link where the visualizations are also available, as i mentioned above.


<p align="right">(<a href="#readme-top">back to top</a>)</p>
