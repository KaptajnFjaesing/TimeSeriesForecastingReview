# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:28:52 2024

@author: petersen.jonas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

stacked_forecasts_statespace = pd.read_pickle('./data/results/stacked_forecasts_statespace.pkl')
stacked_forecasts_exponential_smoothing = pd.read_pickle('./data/results/stacked_forecasts_exponential_smoothing.pkl')
stacked_forecasts_static_mean_profile = pd.read_pickle('./data/results/stacked_forecasts_static_mean_profile.pkl')
stacked_forecasts_rolling_mean_profile = pd.read_pickle('./data/results/stacked_forecasts_rolling_mean_profile.pkl')
stacked_forecasts_sorcerer = pd.read_pickle('./data/results/stacked_forecasts_sorcerer.pkl')
stacked_forecasts_light_gbm = pd.read_pickle('./data/results/stacked_forecasts_light_gbm.pkl')

abs_mean_gradient_training_data = pd.read_pickle('./data/results/abs_mean_gradient_training_data.pkl')

list_of_models_forecasts = [
    stacked_forecasts_statespace,
    stacked_forecasts_exponential_smoothing,
    stacked_forecasts_static_mean_profile,
    stacked_forecasts_rolling_mean_profile,
    stacked_forecasts_sorcerer,
    stacked_forecasts_light_gbm
    ]
forecast_model_names = [
    "SARIMA",
    "Exponential Smoothing",
    "Static Mean Profile",
    "Rolling Mean Profile",
    "Sorcerer",
    "Light GBM"
    ]
colors = [
    'tab:blue',
    'tab:red',
    'tab:green',
    'tab:orange',
    'tab:cyan',
    'tab:brown'
    ]



#%%

time_series_titles = stacked_forecasts_statespace.columns
number_of_time_series = stacked_forecasts_statespace.shape[1]

n_cols = 2  # We want 2 columns
n_rows = int(np.ceil(number_of_time_series / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

for j in range(len(list_of_models_forecasts)):

    mean_df = list_of_models_forecasts[j].groupby(list_of_models_forecasts[j].index).mean()
    std_df = list_of_models_forecasts[j].groupby(list_of_models_forecasts[j].index).std()
    
    # Loop through each column to plot
    for i in range(len(mean_df.columns)):
        ax = axs[i]  # Get the correct subplot
        ax.set_title(time_series_titles[i])
        y_mean = mean_df[mean_df.columns[i]]
        y_std = std_df[mean_df.columns[i]]
        ax.fill_between(mean_df.index, y_mean - y_std, y_mean + y_std, alpha=0.4, color = colors[j], label = forecast_model_names[j])  # Shaded region
        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        ax.grid(True)
        ax.legend()
    
    # Hide any remaining empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])  # Remove unused axes to clean up the figure


# %%

n_cols = 2  # We want 2 columns
n_rows = int(np.ceil(number_of_time_series / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()
for j in range(len(list_of_models_forecasts)):
    mean_df = list_of_models_forecasts[j].abs().groupby(list_of_models_forecasts[j].index).mean()
    # Loop through each column to plot
    for i in range(len(mean_df.columns)):
        ax = axs[i]  # Get the correct subplot
        ax.set_title(time_series_titles[i])
        y_mean = mean_df[mean_df.columns[i]]
        mean_grad = abs_mean_gradient_training_data[mean_df.columns[i]]
        ax.plot(y_mean/mean_grad, color = colors[j], label = forecast_model_names[j])
        ax.set_xlabel('Date')
        ax.set_ylabel('MASE')
        ax.grid(True)
        ax.legend()

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])  # Remove unused axes to clean up the figure

# %% MASE plot


MASE_std_over_time_series = np.zeros((len(list_of_models_forecasts),max(stacked_forecasts_statespace.index)+1))
MASE_averaged_over_time_series = np.zeros((len(list_of_models_forecasts),max(stacked_forecasts_statespace.index)+1))
for j in range(len(list_of_models_forecasts)):
    average_abs_residual = list_of_models_forecasts[j].abs().groupby(list_of_models_forecasts[j].index).mean() # averaged over rolls
    time_series_columns = average_abs_residual.columns
    MASE = average_abs_residual/abs_mean_gradient_training_data
    MASE_averaged_over_time_series[j] = MASE.mean(axis = 1)
    MASE_std_over_time_series[j] = MASE.std(axis = 1)


plt.figure(figsize = (8,5))
for j in range(len(list_of_models_forecasts)):
    plt.plot(MASE_averaged_over_time_series[j], color = colors[j], label = forecast_model_names[j])
    
# Customizing the plot style
plt.grid(visible=True, which='both', linewidth=0.6, color='gray', alpha=0.7)
plt.ylabel('Avg MASE over time series', fontsize=14)  # Flipped label
plt.xlabel('Forecast horizon', fontsize=14)  # Flipped label
plt.title('Model Accuracy Comparison',  fontsize=14)
# Rotate y-axis labels by 90 degrees (if necessary)
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
# Customize tick parameters
plt.tick_params(axis='both', which='major', labelsize=12, length=6, width=1, colors='black', grid_color='gray', grid_alpha=0.7)
plt.tight_layout()
plt.legend()

    
plt.figure(figsize = (8,5))
plt.errorbar(
    x = MASE_averaged_over_time_series.mean(axis= 1),  # Original y values plotted on x-axis
    y = forecast_model_names,  # Original x values plotted on y-axis
    xerr=MASE_averaged_over_time_series.std(axis= 1),  # Error now applied to the x-axis
    fmt='o', color='tab:blue', ecolor='tab:blue', elinewidth=2, capsize=3,
    markersize=5, linewidth=1.5, alpha=1
)

plt.ylabel('Model', fontsize=14)  # Flipped label
plt.xlabel('Avg MASE over time series and forecast horizon', fontsize=14)  # Flipped label
plt.title('Model Accuracy Comparison',  fontsize=14)
# Customizing the plot style
plt.grid(visible=True, which='both', linewidth=0.6, color='gray', alpha=0.7)
# Rotate y-axis labels by 90 degrees (if necessary)
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
# Customize tick parameters
plt.tick_params(axis='both', which='major', labelsize=12, length=6, width=1, colors='black', grid_color='gray', grid_alpha=0.7)
plt.tight_layout()
