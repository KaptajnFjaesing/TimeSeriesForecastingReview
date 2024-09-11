# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:28:52 2024

@author: petersen.jonas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

stacked_forecasts_statespace = pd.read_pickle('./data/results/stacked_forecasts_statespace.pkl')

list_of_models_forecasts = [stacked_forecasts_statespace]
colors = ['tab:red']

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
        ax.fill_between(mean_df.index, y_mean - y_std, y_mean + y_std, alpha=0.4, color = colors[j])  # Shaded region
        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        ax.grid(True)
    
    # Hide any remaining empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])  # Remove unused axes to clean up the figure