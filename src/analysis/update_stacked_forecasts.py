# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:46:30 2024

@author: petersen.jonas
"""

from src.utils import (
    generate_mean_profil_stacked_forecast,
    normalized_weekly_store_category_household_sales,
    generate_SSM_stacked_forecast,
    generate_abs_mean_gradient_training_data,
    generate_exponential_smoothing_stacked_forecast
    )

df = normalized_weekly_store_category_household_sales()


# %%

forecast_horizon = 26
simulated_number_of_forecasts = 50
seasonality_period = 52

generate_abs_mean_gradient_training_data(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts
        )

generate_mean_profil_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        seasonality_period = seasonality_period
        )

generate_exponential_smoothing_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        seasonality_period = seasonality_period,
        )

generate_SSM_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        seasonality_period = seasonality_period
        )

