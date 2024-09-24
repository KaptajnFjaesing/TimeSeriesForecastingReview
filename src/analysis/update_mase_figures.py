"""
Created on Mon Sep 23 08:25:47 2024

@author: Jonas Petersen
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.analysis.model_data import model_data

def update_mase_figures():
    # Load the forecast data and gradient
    for model_name, model_info in model_data.items():
        model_data[model_name]['data'] = pd.read_pickle(model_info['file'])
    # Separate the gradient data
    abs_mean_gradient_training_data = model_data['abs_mean_gradient_training_data']['data']
    forecast_models = {k: v for k, v in model_data.items() if k != 'abs_mean_gradient_training_data'}
    print(forecast_models)

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
    plt.savefig(r'.\docs\report\figures\avg_mase_over_time_series.pdf')
    
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
    plt.savefig(r'.\docs\report\figures\avg_mase_over_time_series_and_forecast_horizon.pdf')

update_mase_figures()