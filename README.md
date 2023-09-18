# Forex Forecasting

This repository contains models designed to forecast foreign exchange currencies, using Neural Networks. Two hybrid methods are presented ([SCNN_Model](SCNN_Model) and [es-rnn](es-rnn)) while a purely statistical one ([V-AR-model](V-AR-model)) was used as a benchmark. A detailed analysis of implementations, methodology, and results can be found in the [report](report/report.pdf).

The project was carried out as a key part of the curriculum for the 'Μ401 - Deep Neural Networks' course, as taught at the National and Kapodistrian University of Athens (NKUA), during the Fall of 2022. 


## Project Structure

```bash
$PROJECT_ROOT
¦
+-- SCNN_Model 
¦   # Forecasting using a simple Exponential Smoothing
¦   # and Convolutional Neural Networks
¦
+-- V-AR-model 
¦   # Forecasting using Vector Autoregression
¦   # (used as a benchmark)
¦
+-- es-rnn 
¦   # Forecasting using Holt-Winters and Recurrent Neural Networks
¦
+-- presentation 
¦   # Presentation summarising implementations, results, and conclusions 
¦
+-- report 
    # Comprehensive report detailing implementations, results, and conclusions
```


## Algorithms

Before diving into the individual methodologies, ensure you have the necessary dependencies installed. Each implementation directory ([SCNN_Model](SCNN_Model), [V-AR-model](V-AR-model), [es-rnn](es-rnn)) contains its own `README.md` file, detailing specific requirements and dependencies, so have a look at it before running the respective model.


## Data Source

The data is sourced from the European Central Bank and can be downloaded using a [script](es-rnn/fetch_data.py) of the repo. The dataset contains historical exchange rates of various currencies against the Euro, and is published around 16:00 CET.

These rates represent the official currencies of non-euro area Member States of the European Union and world currencies with the most liquid active spot FX markets. The dataset can be downloaded from [here](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html), and an overview can be seen bellow.

| Date       | USD   | JPY   | BGN   | ... | THB   | ZAR   |
|------------|-------|-------|-------|-----|-------|-------|
| 2023-09-15 | 1.0658| 157.50| 1.9558| ... | 38.145| 20.2968|
| 2023-09-14 | 1.0730| 158.13| 1.9558| ... | 38.387| 20.3109|
| 2023-09-13 | 1.0733| 158.28| 1.9558| ... | 38.397| 20.3300|
| ...        | ...   | ...   | ...   | ... | ...   | ...   |
| 1999-01-04 | 1.1789| 133.73| NaN   | ... | NaN   | 6.9358 |


## Authors

* **Darmanis Michael** - [mdarm](https://github.com/mdarm)
* **Efstathios Kotsis** - [staks1](https://github.com/staks1)
* **Vasilis Venieris** - [vasilisvenieris](https://github.com/vasilisvenieris)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
