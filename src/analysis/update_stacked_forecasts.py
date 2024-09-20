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
    generate_exponential_smoothing_stacked_forecast,
    generate_sorcerer_stacked_forecast
    )

df = normalized_weekly_store_category_household_sales()

forecast_horizon = 26
simulated_number_of_forecasts = 50
number_of_weeks_in_a_year = 52.1429

# %%

generate_abs_mean_gradient_training_data(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts
        )

generate_mean_profil_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        seasonality_period = number_of_weeks_in_a_year
        )

generate_exponential_smoothing_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        seasonality_period = number_of_weeks_in_a_year,
        )

generate_SSM_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        seasonality_period = number_of_weeks_in_a_year
        )

#%%



sampler_config = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "MAP",
    "discard_tuned_samples": True
}

model_config = {
    "number_of_individual_trend_changepoints": 10,
    "delta_mu_prior": 0,
    "delta_b_prior": 0.1,
    "m_sigma_prior": 0.2,
    "k_sigma_prior": 0.2,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior" : 1,
    "precision_target_distribution_prior_alpha": 2,
    "precision_target_distribution_prior_beta": 0.1,
    "prior_probability_shared_seasonality_alpha": 1,
    "prior_probability_shared_seasonality_beta": 1,
    "individual_fourier_terms": [
        {'seasonality_period_baseline': number_of_weeks_in_a_year,'number_of_fourier_components': 10}
    ],
    "shared_fourier_terms": [
        {'seasonality_period_baseline': number_of_weeks_in_a_year,'number_of_fourier_components': 5},
        {'seasonality_period_baseline': number_of_weeks_in_a_year/4,'number_of_fourier_components': 3},
        {'seasonality_period_baseline': number_of_weeks_in_a_year/12,'number_of_fourier_components': 3},
    ]
}

generate_sorcerer_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        sampler_config = sampler_config,
        model_config = model_config
        )
