# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:34:56 2024

@author: petersen.jonas
"""

from src.load_data import normalized_weekly_store_category_household_sales
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = normalized_weekly_store_category_household_sales()


#%%


import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


def exponential_smoothing(
        df: pd.DataFrame,
        time_series_column: str,
        seasonality_period: int,
        forecast_periods: int
        ):
    
    df_temp = df.copy()
    df_temp.set_index('date', inplace=True)

    # Extract the time series data
    time_series = df_temp[time_series_column]
    
    # Fit Exponential Smoothing Model
    model = ExponentialSmoothing(
        time_series, 
        seasonal='add',  # Use 'add' for additive seasonality or 'mul' for multiplicative seasonality
        trend='add',      # Use 'add' for additive trend or 'mul' for multiplicative trend
        seasonal_periods=seasonality_period  # Adjust for the frequency of your seasonal period (e.g., 12 for monthly data)
    )
    fit_model = model.fit()
    return fit_model.forecast(steps=forecast_periods)


#%%

seasonality_period = 52
harmonics = 10
autoregressive = 2
min_forecast_horizon = 26
max_forecast_horizon = 52
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

model_forecasts = []

for forecast_horizon in tqdm(range(min_forecast_horizon,52+1)):

    training_data = df.iloc[:-forecast_horizon].reset_index()
    y_train_min = training_data[unnormalized_column_group].min()
    y_train_max = training_data[unnormalized_column_group].max()
    
    y_train = (training_data[unnormalized_column_group]-y_train_min)/(y_train_max-y_train_min)
    y_train['date'] = training_data['date']
    model_denormalized = []
    
    time_series_columns = [x for x in y_train.columns if not 'date' in x]
    for i in range(len(time_series_columns)):
        model_results = exponential_smoothing(
                df = y_train,
                time_series_column = time_series_columns.columns[i],
                seasonality_period = seasonality_period,
                forecast_periods = forecast_horizon
                )
        model_denormalized.append(model_results['mean']*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]])

    model_forecasts.append(model_denormalized)


forecast = exponential_smoothing(
        df = df,
        time_series_column = "data",
        seasonality_period = 52,
        forecast_periods = 6
        )

# Print the forecast
print(forecast)


# %%

seasonality_period = 52
harmonics = 10
autoregressive = 2
min_forecast_horizon = 26
max_forecast_horizon = 52
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

model_forecasts = []

forecast_horizon = min_forecast_horizon

training_data = df.iloc[:-forecast_horizon].reset_index()
y_train_min = training_data[unnormalized_column_group].min()
y_train_max = training_data[unnormalized_column_group].max()

y_train = (training_data[unnormalized_column_group]-y_train_min)/(y_train_max-y_train_min)
y_train['date'] = training_data['date']
model_denormalized = []

time_series_columns = [x for x in y_train.columns if not 'date' in x]
i = 0
model_results = exponential_smoothing(
        df = y_train,
        time_series_column = time_series_columns.columns[i],
        seasonality_period = seasonality_period,
        forecast_periods = forecast_horizon
        )