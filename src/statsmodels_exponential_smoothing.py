# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:34:47 2024

@author: petersen.jonas
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import matplotlib.pyplot as plt

dates = pd.date_range(start='2020-01-01', periods=36, freq='M')  # Monthly data


df = pd.DataFrame({
    'date': dates,  # Monthly data
    'data': np.random.normal(10, 2, size=len(dates))  # Example data with a trend and seasonality
})
df['date'] = pd.to_datetime(df['date'])

plt.figure()
plt.plot(df['date'],df['data'])

# %%
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


forecast = exponential_smoothing(
        df = df,
        time_series_column = "data",
        seasonality_period = 12,
        forecast_periods = 6
        )

# Print the forecast
print(forecast)


# %%

plt.figure()
plt.plot(df['date'],df['data'])
plt.plot(forecast)
