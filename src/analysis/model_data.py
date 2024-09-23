"""
Created on Mon Sep 23 08:31:10 2024

@author: Jonas Petersen
"""

model_data = {
    'Static Mean Profile': {
        'file': './data/results/stacked_forecasts_static_mean.pkl',
        'color': 'tab:blue'
    },
    'Rolling Mean Profile': {
        'file': './data/results/stacked_forecasts_rolling_mean.pkl',
        'color': 'tab:red'
    },
    'Exponential Smoothing': {
        'file': './data/results/stacked_forecasts_exponential_smoothing.pkl',
        'color': 'tab:green'
    },
    'SSM': {
        'file': './data/results/stacked_forecasts_statespace.pkl',
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
        'file': './data/results/stacked_forecasts_light_gbm.pkl',
        'color': 'tab:gray'
    },
    'Light GBM w. sklearn': {
        'file': './data/results/stacked_forecasts_light_gbm_w_sklearn.pkl',
        'color': 'tab:olive'
    },
    'abs_mean_gradient_training_data': {
        'file': './data/results/abs_mean_gradient_training_data.pkl'
    }
}