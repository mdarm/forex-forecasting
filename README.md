# Forex Forecasting using Neural Networks 

## Purpose

The purpose of this codebase is to forecast foreign exchange rates (Forex) using Neural Networks. 

## Data Source

The data is sourced from the European Central Bank and can be downloaded using the script in this repository. The dataset contains historical exchange rates of various currencies against the Euro, and is published around 16:00 CET.

These rates represent the official currencies of non-euro area Member States of the European Union and world currencies with the most liquid active spot FX markets. The dataset can be downloaded from [here](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html), and an overview can be seen bellow.

| Date       | USD   | JPY   | BGN   | ... | THB   | ZAR   |
|------------|-------|-------|-------|-----|-------|-------|
| 2023-09-15 | 1.0658| 157.50| 1.9558| ... | 38.145| 20.2968|
| 2023-09-14 | 1.0730| 158.13| 1.9558| ... | 38.387| 20.3109|
| 2023-09-13 | 1.0733| 158.28| 1.9558| ... | 38.397| 20.3300|
| ...        | ...   | ...   | ...   | ... | ...   | ...   |
| 1999-01-04 | 1.1789| 133.73| NaN   | ... | NaN   | 6.9358 |

## Steps

1. Data is first downloaded from the European Central Bank's website.
2. Preprocessing steps are performed to clean the data and make it suitable for training.
3. Various forecasting models from the PyTorch Forecasting library are then applied to the cleaned data.

## Requirements

Install the requirements by simpy typing:

```bash
pip install -r requirements.txt
```

## Authors

* **Darmanis Michael** - [mdarm](https://github.com/mdarm)
* **Efstathios Kotsis** - [staks1](https://github.com/staks1)
* **Vasilis Venieris** - [vasilisvenieris](https://github.com/vasilisvenieris)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
