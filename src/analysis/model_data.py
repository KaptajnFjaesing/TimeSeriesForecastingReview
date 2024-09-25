"""
Created on Mon Sep 23 08:31:10 2024

@author: Jonas Petersen
"""

model_data = {
    'Static Mean Profile': {
        'file': './data/results/stacked_residuals_static_mean.pkl',
        'color': 'tab:blue'
    },
    'Rolling Mean Profile': {
        'file': './data/results/stacked_forecasts_rolling_mean.pkl',
        'color': 'tab:red'
    },
    'Exponential Smoothing': {
        'file': './data/results/stacked_residuals_exponential_smoothing.pkl',
        'color': 'tab:green'
    },
    'SSM': {
        'file': './data/results/stacked_residuals_statespace.pkl',
        'color': 'tab:cyan'
    },
    'Sorcerer v0.3 (MAP)': {
        'file': './data/results/stacked_forecasts_sorcerer_MAP.pkl',
        'color': 'tab:brown'
    },
    'Sorcerer v0.3 (NUTS)': {
        'file': './data/results/stacked_forecasts_sorcerer_NUTS.pkl',
        'color': 'tab:orange'
    },
    'Light GBM Basic': {
        'file': './data/results/stacked_residuals_lgbm_basic.pkl',
        'color': 'tab:gray'
    },
    'Light GBM w. sklearn': {
        'file': './data/results/stacked_residuals_lgbm_sklearn.pkl',
        'color': 'tab:olive'
    },
    'TLP Regression Model (MAP)': {
        'file': './data/results/stacked_forecasts_tlp_regression_model_MAP.pkl',
        'color': 'tab:purple'
    },
    'TLP Regression Model (NUTS)': {
        'file': './data/results/stacked_forecasts_tlp_regression_model_NUTS.pkl',
        'color': 'tab:pink'
    },
    'Naive Darts Model': {
        'file': './data/results/stacked_residuals_naive_darts.pkl',
        'color': 'black'
    },
    'TiDe Darts Model': {
        'file': './data/results/stacked_residuals_tide_darts.pkl',
        'color': 'gold'
    },
    'Light GBM darts Model': {
        'file': './data/results/stacked_residuals_lgbm_darts.pkl',
        'color': 'yellow'
    },
    'XGBoost Darts Model': {
        'file': './data/results/stacked_residuals_xgboost_darts.pkl',
        'color': 'silver'
    },
    'abs_mean_gradient_training_data': {
        'file': './data/results/abs_mean_gradient_training_data.pkl'
    }
}