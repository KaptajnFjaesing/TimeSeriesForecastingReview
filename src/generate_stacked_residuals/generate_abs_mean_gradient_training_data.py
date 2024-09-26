"""
Created on Wed Sep 25 10:26:41 2024

@author: Jonas Petersen
"""

import pandas as pd

import src.generate_stacked_residuals.global_model_parameters as gmp

def generate_abs_mean_gradient_training_data(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts
        ):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    max_forcast_horizon = forecast_horizon+simulated_number_of_forecasts
    df[time_series_column_group].iloc[:-max_forcast_horizon].diff().dropna().abs().mean(axis = 0).to_pickle('./data/results/abs_mean_gradient_training_data.pkl')    
    print("generate_abs_mean_gradient_training_data completed")

generate_abs_mean_gradient_training_data()