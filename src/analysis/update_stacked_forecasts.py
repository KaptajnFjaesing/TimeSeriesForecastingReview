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
    generate_light_gbm_w_sklearn_stacked_forecast,
    generate_tlp_regression_model_stacked_forecast
    )

from src.analysis.update_mase_figures import update_mase_figures
import argparse

_,df,_ = load_m5_weekly_store_category_sales_data()

forecast_horizon = 26
simulated_number_of_forecasts = 50
number_of_weeks_in_a_year = 52.1429
context_length = 30

sorcerer_MAP_sampler = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "MAP",
    "discard_tuned_samples": True,
    "verbose": False
}

sorcerer_NUTS_sampler = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
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

lgbm_config_basic = {
    'verbose': -1,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 5,
    'learning_rate': 0.1,
    'feature_fraction': 0.9
}

lgbm_config_sklearn = {
    "max_depth": [4, 7, 10],
    "num_leaves": [20, 50],
    "learning_rate": [0.01, 0.05],
    "n_estimators": [100, 400],
    "colsample_bytree": [0.3, 0.5],
    'verbose':[-1]
}

tlp_MAP_sampler = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "MAP",
    "nuts_sampler": "numpyro",
    "verbose": False
}

tlp_NUTS_sampler = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "nuts_sampler": "numpyro",
    "verbose": False
}

base_alpha = 2
base_beta = 0.1

tlp_config = {
    "w_input_alpha_prior": base_alpha,
    "b_input_alpha_prior": base_alpha,
    "w2_alpha_prior": base_alpha,
    "b2_alpha_prior": base_alpha,
    "w_output_alpha_prior": base_alpha,
    "b_output_alpha_prior": base_alpha,
    "precision_alpha_prior": 100,
    "w_input_beta_prior": base_beta,
    "b_input_beta_prior": base_beta,
    "w2_beta_prior": base_beta,
    "b2_beta_prior": base_beta,
    "w_output_beta_prior": base_beta,
    "b_output_beta_prior": base_beta,
    "precision_beta_prior": 0.1,
    "activation": "swish",
    "n_hidden_layer1": 20,
    "n_hidden_layer2": 20
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update model metrics.')
    parser.add_argument('--models', nargs='+', help='List of models to update (e.g., A B C)', required=True)
    
    args = parser.parse_args()
    
    if 'update_figures' in args.models:
        update_mase_figures()
    
    if 'all' in args.models:
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
        
        generate_light_gbm_stacked_forecast(
                df = df,
                simulated_number_of_forecasts = simulated_number_of_forecasts,
                forecast_horizon = forecast_horizon,
                context_length = context_length,
                seasonality_period = number_of_weeks_in_a_year,
                model_config = lgbm_config_basic
                )
        
        generate_light_gbm_w_sklearn_stacked_forecast(
                df = df,
                simulated_number_of_forecasts = simulated_number_of_forecasts,
                forecast_horizon = forecast_horizon,
                model_config = lgbm_config_sklearn
                )

        generate_sorcerer_stacked_forecast(
                df = df,
                forecast_horizon = forecast_horizon,
                simulated_number_of_forecasts = simulated_number_of_forecasts,
                sampler_config = sorcerer_MAP_sampler,
                model_config = sorcerer_config
                )

        generate_sorcerer_stacked_forecast(
                df = df,
                forecast_horizon = forecast_horizon,
                simulated_number_of_forecasts = simulated_number_of_forecasts,
                sampler_config = sorcerer_NUTS_sampler,
                model_config = sorcerer_config
                )
        
        generate_tlp_regression_model_stacked_forecast(
                df = df,
                forecast_horizon = forecast_horizon,
                simulated_number_of_forecasts = simulated_number_of_forecasts,
                sampler_config = tlp_MAP_sampler,
                model_config = tlp_config,
                )
        
        generate_tlp_regression_model_stacked_forecast(
                df = df,
                forecast_horizon = forecast_horizon,
                simulated_number_of_forecasts = simulated_number_of_forecasts,
                sampler_config = tlp_NUTS_sampler,
                model_config = tlp_config,
                )
        
    else:
        
        if 'tlp' in args.models:
            generate_tlp_regression_model_stacked_forecast(
                    df = df,
                    forecast_horizon = forecast_horizon,
                    simulated_number_of_forecasts = simulated_number_of_forecasts,
                    sampler_config = tlp_MAP_sampler,
                    model_config = tlp_config,
                    )
            
            generate_tlp_regression_model_stacked_forecast(
                    df = df,
                    forecast_horizon = forecast_horizon,
                    simulated_number_of_forecasts = simulated_number_of_forecasts,
                    sampler_config = tlp_NUTS_sampler,
                    model_config = tlp_config,
                    )
        if 'sorcerer' in args.models:
            generate_sorcerer_stacked_forecast(
                    df = df,
                    forecast_horizon = forecast_horizon,
                    simulated_number_of_forecasts = simulated_number_of_forecasts,
                    sampler_config = sorcerer_MAP_sampler,
                    model_config = sorcerer_config
                    )

            generate_sorcerer_stacked_forecast(
                    df = df,
                    forecast_horizon = forecast_horizon,
                    simulated_number_of_forecasts = simulated_number_of_forecasts,
                    sampler_config = sorcerer_NUTS_sampler,
                    model_config = sorcerer_config
                    )
        if 'lgbm' in args.models:
            generate_light_gbm_stacked_forecast(
                    df = df,
                    simulated_number_of_forecasts = simulated_number_of_forecasts,
                    forecast_horizon = forecast_horizon,
                    context_length = context_length,
                    seasonality_period = number_of_weeks_in_a_year,
                    model_config = lgbm_config_basic
                    )
            
            generate_light_gbm_w_sklearn_stacked_forecast(
                    df = df,
                    simulated_number_of_forecasts = simulated_number_of_forecasts,
                    forecast_horizon = forecast_horizon,
                    model_config = lgbm_config_sklearn
                    )
