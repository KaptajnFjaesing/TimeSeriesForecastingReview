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
from src.utils import compute_residuals

def generate_mean_profile(df, seasonality_period, normalized_column_group):
    condition = df['date'].dt.year < df['date'].dt.year.max()
    training_data = df[condition].reset_index()
    df_melted = training_data.melt(
        id_vars='week',
        value_vars= normalized_column_group,
        var_name='Variable',
        value_name='Value'
        )
    return np.array([df_melted['Value'][df_melted['week'] == week].values.mean() for week in range(1,seasonality_period+1)])

def generate_mean_profil_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        seasonality_period: int,
        file_name: str,
        MA_window: int
        ):
    
    normalized_column_group = [x for x in df.columns if '_normalized' in x ]
    unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
 
    model_forecasts_rolling = []
    model_forecasts_static = []
    for fh in range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts+1):
        training_data = df.iloc[:-fh].reset_index()
        mean_profile = generate_mean_profile(
            df = training_data,
            seasonality_period = seasonality_period,
            normalized_column_group = normalized_column_group
            )
        
        projected_scales = training_data[unnormalized_column_group].iloc[-MA_window:].mean(axis = 0)
        week_indices_in_forecast = [int(week - 1) for week in df.iloc[-fh:]['week'].values]
        profile_subset = mean_profile[week_indices_in_forecast]/mean_profile[week_indices_in_forecast][0]
        model_forecasts_rolling.append([pd.Series(row) for row in np.outer(projected_scales,profile_subset)])
        
        
        projected_scales = []
        for col in unnormalized_column_group:
            yearly_averages = training_data[['year']+[col]].groupby('year').mean()
            mean_grad = training_data[['year']+[col]].groupby('year').mean().diff().dropna().mean().values[0]
            projected_scales.append(yearly_averages.values[-1,0]+mean_grad)
        projected_scales = pd.DataFrame(data = projected_scales).T
        projected_scales.columns = unnormalized_column_group
        
        profile_subset = mean_profile[week_indices_in_forecast]
        model_forecasts_static.append([pd.Series(row) for row in np.outer(projected_scales,profile_subset)])
    
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    
    compute_residuals(
             model_forecasts = model_forecasts_rolling,
             test_data = test_data[unnormalized_column_group],
             min_forecast_horizon = forecast_horizon
             ).to_pickle(f'./data/results/stacked_forecasts_rolling_{file_name}.pkl')    
    
    compute_residuals(
             model_forecasts = model_forecasts_static,
             test_data = test_data[unnormalized_column_group],
             min_forecast_horizon = forecast_horizon
             ).to_pickle(f'./data/results/stacked_forecasts_static_{file_name}.pkl')
    


seasonality_period = 52
forecast_horizon = 26
MA_window = 10
file_name = "mean"
simulated_number_of_forecasts = 40


generate_mean_profil_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        seasonality_period = seasonality_period,
        file_name = file_name,
        MA_window = MA_window
        )
