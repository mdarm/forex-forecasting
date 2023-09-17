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


## Built With

The code and models were built using :
* python
* Keras
* Tensorflow


<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
