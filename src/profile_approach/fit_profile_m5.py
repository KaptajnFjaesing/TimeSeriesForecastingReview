# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:18:57 2024

@author: petersen.jonas
"""

from src.utils import normalized_weekly_store_category_household_sales

df = normalized_weekly_store_category_household_sales()

# %%
import numpy as np
from tqdm import tqdm
import pandas as pd


seasonality_period = 52
min_forecast_horizon = 26
max_forecast_horizon = 52
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
normalized_column_group = [x for x in df.columns if '_normalized' in x ]

model_forecasts = []

for forecast_horizon in tqdm(range(min_forecast_horizon,52+1)):

    training_data = df.iloc[:-forecast_horizon].reset_index()
    y_train_min = training_data[unnormalized_column_group].min()
    y_train_max = training_data[unnormalized_column_group].max()
    
    y_train = (training_data[unnormalized_column_group]-y_train_min)/(y_train_max-y_train_min)
    y_train['date'] = training_data['date']
    
    df_melted = training_data.melt(
        id_vars='week',
        value_vars= normalized_column_group,
        var_name='Variable',
        value_name='Value'
        )
    mean_profile = [df_melted['Value'][df_melted['week'] == week].values.mean() for week in range(1,seasonality_period+1)]
    
    projected_scales = []
    for col in unnormalized_column_group:
        yearly_averages = training_data[['year']+[col]].groupby('year').mean()
        mean_grad = training_data[['year']+[col]].groupby('year').mean().diff().dropna().mean().values[0]
        projected_scales.append(yearly_averages.values[-1,0]+mean_grad)
    projected_scales = np.array(projected_scales)
    model_forecasts.append([pd.Series(row) for row in np.outer(projected_scales,mean_profile)])


#%%

from src.utils import compute_residuals

test_data = df.iloc[-max_forecast_horizon:].reset_index(drop = True)

stacked = compute_residuals(
         model_forecasts = model_forecasts,
         test_data = test_data[unnormalized_column_group],
         min_forecast_horizon = min_forecast_horizon
         )

stacked.to_pickle('./data/results/stacked_forecasts_mean_profile.pkl')
