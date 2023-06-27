# LazyForecast
LazyForecast is a Python library for performing univariate time series analysis using a lazy forecasting approach. This approach is designed to provide quick and simple forecasting models without requiring extensive configuration or parameter tuning.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Output](#output)

## Installation
You can install LazyForecast using pip:
```bash
pip install lazyforecast
```

## Features
- LazyForecasting automatically selects the best model based on the characteristics of the input time series.
- It supports univariate time series analysis.
- LazyForecasting provides functions for data preprocessing, model training, forecasting, and evaluation.
- It includes various popular forecasting models such as Auto ARIMA, Vanilla LSTM, and RNN.

## Usage
Here's an example of how to use the LazyForecast library to forecast stock prices using historical data:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import LazyForecast as lf

# Set the number of periods and steps for forecasting
n_periods = 50
n_steps = 5

# Create an instance of LazyForecast
ts = lf.LazyForecast(n_periods=n_periods, n_steps=n_steps, n_members=5)

# Specify the start and end dates for the data
start_date = '2021-01-01'
end_date = '2022-12-31'

# Fetch the historical stock data for Google
df = yf.download('GOOGL', start=start_date, end=end_date)

# Reset the index of the DataFrame
df.reset_index(level=0, inplace=True)

# Fit the data to the LazyForecast model and obtain evaluation metrics, forecasts, and confidence intervals
eval_df, fc, confint = ts.fit(df, x_axis='Date', y_axis='Close')

# Print the evaluation metrics for each model
print(eval_df)
```

## Output

### Auto-Regressive Integrated Moving Average (ARIMA)
![Imgur](https://i.imgur.com/ML90i3D.png)

### Multi-Layer Perceptron (MLP)
![Alt text](https://i.imgur.com/M2s8olH.png)

### Vanilla Long Short-Term Memory (LSTM)
![Alt text](https://i.imgur.com/R8KDORF.png)

### Stacked LSTM
![Alt text](https://i.imgur.com/lnW9IZd.png)

### Bi-directional LSTM
![Alt text](https://i.imgur.com/lrjYoFm.png)

### Recurrent Neural Network (RNN)
![Alt text](https://i.imgur.com/k4VQ6lH.png)

### Gated Recurrent Unit (GRU)
![Alt text](https://i.imgur.com/8hiG805.png)

### Evaluation Table
  
| model              |      mda |    rmse |      mape |       R2 |     mae |     corr |
| :----------------- | -------: | ------: | --------: | -------: | ------: | -------: |
| ARIMA              |  0.55102 | 2.62852 | 0.0208377 |  0.72456 |  1.9532 | 0.863465 |
| GRU                |  0.55102 |  2.6993 | 0.0216135 | 0.667912 | 2.03898 |  0.85146 |
| BIDIRECTIONAL LSTM |  0.55102 | 2.72504 | 0.0220456 | 0.639895 | 2.08601 | 0.851686 |
| VANILLA LSTM       |  0.55102 | 2.79671 | 0.0225355 | 0.630066 | 2.13659 | 0.848691 |
| STACKED LSTM       | 0.510204 | 2.90771 | 0.0233817 | 0.581736 | 2.22243 | 0.851591 |
| MLP                | 0.510204 | 7.57871 | 0.0664394 |  -4.6669 | 6.66732 | 0.698219 |
| RNN                | 0.469388 | 2.96933 | 0.0240381 | 0.609076 | 2.27333 | 0.821808 |
