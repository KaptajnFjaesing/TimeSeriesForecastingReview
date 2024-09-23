# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:46:30 2024

@author: petersen.jonas
"""

from src.utils import (
    load_m5_weekly_store_category_sales_data,
    generate_mean_profil_stacked_forecast,
    generate_SSM_stacked_forecast,
    generate_abs_mean_gradient_training_data,
    generate_exponential_smoothing_stacked_forecast,
    generate_sorcerer_stacked_forecast,
    generate_light_gbm_stacked_forecast,
    generate_light_gbm_w_sklearn_stacked_forecast
    )

from src.analysis.update_mase_figures import update_mase_figures

_,df,_ = load_m5_weekly_store_category_sales_data()

forecast_horizon = 26
simulated_number_of_forecasts = 50
number_of_weeks_in_a_year = 52.1429
context_length = 30

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

# %% Light GBM basic

lgbm_config_basic = {
    'verbose': -1,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 5,
    'learning_rate': 0.1,
    'feature_fraction': 0.9
}

generate_light_gbm_stacked_forecast(
        df = df,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        forecast_horizon = forecast_horizon,
        context_length = context_length,
        seasonality_period = number_of_weeks_in_a_year,
        model_config = lgbm_config_basic
        )


#%% Light GBM w sklearn

lgbm_config_sklearn = {
    "max_depth": [4, 7, 10],
    "num_leaves": [20, 50],
    "learning_rate": [0.01, 0.05],
    "n_estimators": [100, 400],
    "colsample_bytree": [0.3, 0.5],
    'verbose':[-1]
}

generate_light_gbm_w_sklearn_stacked_forecast(
        df = df,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        forecast_horizon = forecast_horizon,
        model_config = lgbm_config_sklearn
        )

#%% Sorcerer MAP

sorcerer_sampler = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "MAP",
    "discard_tuned_samples": True,
    "verbose": False
}

sorcerer_config = {
    "number_of_individual_trend_changepoints": 10,
    "delta_mu_prior": 0,
    "delta_b_prior": 0.1,
    "m_sigma_prior": 0.2,
    "k_sigma_prior": 0.2,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior" : 1,
    "precision_target_distribution_prior_alpha": 100,
    "precision_target_distribution_prior_beta": 1,
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
        sampler_config = sorcerer_sampler,
        model_config = sorcerer_config
        )

#%% Sorcerer NUTS

sorcerer_sampler = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "discard_tuned_samples": True,
    "verbose": False
}

generate_sorcerer_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        sampler_config = sorcerer_sampler,
        model_config = sorcerer_config
        )

#%% Update mase figures

update_mase_figures()