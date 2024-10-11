"""
Created on Mon Sep 23 08:25:47 2024

@author: Jonas Petersen
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the directory containing the results
results_dir = 'data/results/'

# Initialize the model data dictionary
model_data = {}

# Define a mapping of substrings to human-readable names
name_mapping = {
    'static_mean': 'Static Mean Profile',
    'rolling_mean': 'Rolling Mean Profile',
    'exponential_smoothing': 'Holt-Winters',
    'statespace': 'SSM',
    'sorcerer_MAP': 'Sorcerer v0.4.1 (MAP)',
    'sorcerer_NUTS': 'Sorcerer v0.4.1 (NUTS)',
    'lgbm_basic': 'Light GBM Basic',
    'lgbm_sklearn': 'Light GBM w. sklearn',
    'tlp_MAP': 'TLP Regression Model (MAP)',
    'tlp_NUTS': 'TLP Regression Model (NUTS)',
    'naive_darts': 'Naive Darts Model',
    'tide_darts': 'TiDe Darts Model',
    'tft_darts': 'TFT Darts Model',
    'lgbm_darts': 'Light GBM darts Model',
    'lgbm_feature_darts': 'Light GBM Feature Darts Model',
    'xgboost_darts': 'XGBoost Darts Model',
    'abs_mean_gradient_training_data': 'abs_mean_gradient_training_data',
    'deepar': 'DeepAR GluonTS',
    'naive_seasonal_darts': 'Naive Seasonal Darts',
    'climatological_darts': 'Climatological Darts'
}

# Define a color mapping (optional)
color_mapping = {
    'static_mean': 'tab:blue',
    'rolling_mean': 'tab:red',
    'exponential_smoothing': 'tab:green',
    'statespace': 'tab:cyan',
    'sorcerer_MAP': 'tab:brown',
    'sorcerer_NUTS': 'tab:orange',
    'lgbm_basic': 'tab:gray',
    'lgbm_sklearn': 'tab:olive',
    'tlp_MAP': 'tab:purple',
    'tlp_NUTS': 'tab:pink',
    'naive_darts': 'black',
    'tide_darts': 'gold',
    'tft_darts': 'darkred',
    'lgbm_darts': 'yellow',
    'lgbm_feature_darts': 'darkorange',
    'xgboost_darts': 'silver',
    'deepar': 'tomato',
    'naive_seasonal_darts': 'darkgreen',
    'climatological_darts': 'darkblue'
}

# Iterate through files in the results directory
for file_name in os.listdir(results_dir):
    if file_name.endswith('.pkl'):
        # Determine the name based on substrings
        for key in name_mapping.keys():
            if key in file_name:
                model_data[name_mapping[key]] = {
                    'file': os.path.join(results_dir, file_name),
                    'color': color_mapping.get(key)  # Optional: Get the color if available
                }
                break  # Exit the loop once a match is found


# Load the forecast data and gradient
for model_name, model_info in model_data.items():
    model_data[model_name]['data'] = pd.read_pickle(model_info['file'])
# Separate the gradient data
abs_mean_gradient_training_data = model_data['abs_mean_gradient_training_data']['data']
forecast_models = {k: v for k, v in model_data.items() if k != 'abs_mean_gradient_training_data'}

# Initialize MASE arrays
horizon = max(forecast_models['Static Mean Profile']['data'].index) + 1
MASE_std_over_time_series = np.zeros((len(forecast_models), horizon))
MASE_averaged_over_time_series = np.zeros((len(forecast_models), horizon))

# Compute MASE for each model
for j, (model_name, model_info) in enumerate(forecast_models.items()):
    forecast_data = model_info['data']
    average_abs_residual = forecast_data.abs().groupby(forecast_data.index).mean()
    MASE = average_abs_residual / abs_mean_gradient_training_data
    MASE_averaged_over_time_series[j] = MASE.mean(axis=1)
    MASE_std_over_time_series[j] = MASE.std(axis=1)

# Plot average MASE over forecast horizon
plt.figure(figsize=(10, 10))
for model_name, model_info in forecast_models.items():
    plt.plot(MASE_averaged_over_time_series[list(forecast_models.keys()).index(model_name)], 
             color=model_info['color'], label=model_name)
plt.grid(visible=True, which='both', linewidth=0.6, color='gray', alpha=0.7)
plt.ylabel('Avg MASE over time series', fontsize=14)
plt.xlabel('Forecast horizon', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.tight_layout()
# plt.savefig(r'.\docs\report\figures\avg_mase_over_time_series.pdf')

# Plot error bars for MASE across models
plt.figure(figsize=(8, 5))
plt.errorbar(
    x=MASE_averaged_over_time_series.mean(axis=1),
    y=list(forecast_models.keys()),
    xerr=MASE_averaged_over_time_series.std(axis=1),
    fmt='o', color='tab:blue', ecolor='tab:blue', elinewidth=2, capsize=3,
    markersize=5, linewidth=1.5, alpha=1
)
plt.ylabel('Model', fontsize=14)
plt.xlabel('Avg MASE over time series and forecast horizon', fontsize=14)
plt.grid(visible=True, which='both', linewidth=0.6, color='gray', alpha=0.7)
plt.tight_layout()
# plt.savefig(r'.\docs\report\figures\avg_mase_over_time_series_and_forecast_horizon.pdf')

