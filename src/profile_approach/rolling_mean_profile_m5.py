# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:27:34 2024

@author: petersen.jonas
"""

from src.utils import normalized_weekly_store_category_household_sales

df = normalized_weekly_store_category_household_sales()

# %%
import numpy as np
import pandas as pd

seasonality_period = 52
min_forecast_horizon = 26
max_forecast_horizon = 52
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
normalized_column_group = [x for x in df.columns if '_normalized' in x ]

# Generate mean_profile
condition = df['date'].dt.year < df['date'].dt.year.max()
training_data = df[condition].reset_index()
df_melted = training_data.melt(
    id_vars='week',
    value_vars= normalized_column_group,
    var_name='Variable',
    value_name='Value'
    )
mean_profile = np.array([df_melted['Value'][df_melted['week'] == week].values.mean() for week in range(1,seasonality_period+1)])



# %%

# Pick the part of the static forecast "np.outer(projected_scales,mean_profile)" that is relevant
model_forecasts = []
for forecast_horizon in range(min_forecast_horizon,max_forecast_horizon+1):
    training_data = df.iloc[:-forecast_horizon].reset_index()
    
    projected_scales = training_data[unnormalized_column_group].iloc[-10:].mean(axis = 0)
    
    
    week_indices_in_forecast = [int(week - 1) for week in df.iloc[-forecast_horizon:]['week'].values]
    profile_subset = mean_profile[week_indices_in_forecast]/mean_profile[week_indices_in_forecast][0]
    model_forecasts.append([pd.Series(row) for row in np.outer(projected_scales,profile_subset)])

#%%

from src.utils import compute_residuals

test_data = df.iloc[-max_forecast_horizon:].reset_index(drop = True)

stacked = compute_residuals(
         model_forecasts = model_forecasts,
         test_data = test_data[unnormalized_column_group],
         min_forecast_horizon = min_forecast_horizon
         )

stacked.to_pickle('./data/results/stacked_forecasts_rolling_mean_profile.pkl')
