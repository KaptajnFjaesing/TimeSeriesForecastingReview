"""
Created on Mon Sep 23 08:25:47 2024

@author: Jonas Petersen
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import matplotlib.table as tbl
import src.generate_stacked_residuals.global_model_parameters as gmp

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
    'sorcerer_MAP': 'Sorcerer v0.4.4 (MAP)',
    'sorcerer_NUTS': 'Sorcerer v0.4.4 (NUTS)',
    'lgbm_basic': 'Light GBM Basic',
    'lgbm_sklearn': 'Light GBM w. sklearn',
    'tlp_MAP': 'TLP Regression (MAP)',
    'tlp_NUTS': 'TLP Regression (NUTS)',
    'naive_darts': 'Naive Drift Darts',
    'tide_darts': 'TiDe Darts',
    'tft_darts': 'TFT Darts',
    'lgbm_darts': 'Light GBM darts',
    'lgbm_feature_darts': 'Light GBM Feature Darts',
    'xgboost_darts': 'XGBoost Darts',
    'abs_mean_gradient_training_data': 'abs_mean_gradient_training_data',
    'deepar': 'DeepAR GluonTS',
    'naive_seasonal_darts': 'Naive Seasonal Darts',
    'climatological_darts': 'Climatological Darts',
    'naive_no_drift': 'Naive Darts'
}


# Define a color mapping (optional)
color_mapping = {
    'static_mean': '#1f77b4',  # Blue
    'rolling_mean': '#ff7f0e',  # Orange
    'exponential_smoothing': '#2ca02c',  # Green
    'statespace': '#d62728',  # Red
    'sorcerer_MAP': '#9467bd',  # Purple
    'sorcerer_NUTS': '#8c564b',  # Brown
    'lgbm_basic': '#e377c2',  # Pink
    'lgbm_sklearn': '#7f7f7f',  # Gray
    'tlp_MAP': '#bcbd22',  # Yellow-green
    'tlp_NUTS': '#17becf',  # Cyan
    'naive_darts': '#000000',  # Black
    'tide_darts': '#ffbb78',  # Light Orange
    'tft_darts': '#ff9896',  # Light Red
    'lgbm_darts': '#98df8a',  # Light Green
    'lgbm_feature_darts': '#c5b0d5',  # Light Purple
    'xgboost_darts': '#c49c94',  # Light Brown
    'abs_mean_gradient_training_data': '#aec7e8',  # Light Blue
    'deepar': '#f7b6d2',  # Light Pink
    'naive_seasonal_darts': '#c7c7c7',  # Light Gray
    'climatological_darts': '#dbdb8d',  # Light Yellow-green
    'naive_no_drift': '#ffbb78'  # Light Orange
}

with open(os.path.join(results_dir, 'computation_times.json'), 'r') as f:
    computation_times = json.load(f)

# List of models using parallel computation
parallel_models = ['sorcerer', 'statespace', 'tft', 'tide_feature', 'tlp']

# Adjust computation times for parallel models
for model in parallel_models:
    for key in computation_times.keys():
        if model in key:
            computation_times[key] *= gmp.n_jobs

computation_times['static_mean'] = computation_times['generate_mean_profile_stacked_residuals'] /2
computation_times['rolling_mean'] = computation_times['generate_mean_profile_stacked_residuals'] /2

temp = {}
for rel_key in computation_times.keys():
    for key in name_mapping.keys():
        if key in rel_key:
            temp[rel_key] =  name_mapping[key]
            break  # Exit the loop once a match is found

computation_times_renamed = {
    temp.get(key, key): value for key, value in computation_times.items()
}

# Normalize computation times relative to the slowest model
max_time = max(computation_times.values())
relative_times = {model: time / max_time for model, time in computation_times_renamed.items()}

# Iterate through files in the results directory
for file_name in os.listdir(results_dir):
    if file_name.endswith('.pkl'):
        try:
            relative_time = relative_times[file_name[:-4]]
        except:
            relative_time = np.nan
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
plt.savefig(r'.\figures\avg_mase_over_time_series.png')

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
plt.savefig(r'.\figures\avg_mase_over_time_series_and_forecast_horizon.png')

# Create subplots for each model in three columns
num_models = len(forecast_models)
num_rows = (num_models + 2) // 3  # Calculate the number of rows needed for three columns
fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 2 * num_rows), sharex=True)

# Flatten axes for easier iteration
axes = axes.flatten()

for ax, (model_name, model_info) in zip(axes, forecast_models.items()):
    ax.plot(
        MASE_averaged_over_time_series[list(forecast_models.keys()).index(model_name)],
        color='tab:blue'  # Use a single color for all plots
    )
    ax.set_title(model_name, fontsize=10, loc='left')
    ax.grid(visible=True, which='both', linewidth=0.6, color='gray', alpha=0.7)
    ax.set_ylabel('Avg MASE', fontsize=8)

# Remove unused subplots
for ax in axes[len(forecast_models):]:
    ax.axis('off')

axes[-1].set_xlabel('Forecast horizon', fontsize=10)  # Set x-label only on the last subplot
plt.tight_layout()
plt.savefig(r'.\figures\avg_mase_subplots_three_columns.png')


# Calculate overall performance as the mean of avg MASE over time series and forecast horizon
overall_performance = MASE_averaged_over_time_series.mean(axis=1)

# Prepare data for the table and sort by performance (best on top)
summary_data = sorted(
    [
        [
            model, 
            f"{performance:.2f} ± {MASE_averaged_over_time_series.std(axis=1)[i]:.2f}", 
            f"{relative_times.get(model, np.nan):.2f}"
        ]
        for i, (model, performance) in enumerate(zip(forecast_models.keys(), overall_performance))
    ],
    key=lambda x: float(x[1].split(' ± ')[0]),  # Sort by performance value
    reverse=False  # Best on top
)

# Create a figure for the table
fig, ax = plt.subplots(figsize=(8, len(summary_data) * 0.4))
ax.axis('tight')
ax.axis('off')

# Add the table
table = tbl.Table(ax, bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)

# Add header row
header = ['Model', 'Overall Performance (Avg MASE)', 'Relative Computation Time']
for col, text in enumerate(header):
    table.add_cell(0, col, width=1/len(header), height=0.4, text=text, loc='center', edgecolor='black', facecolor='lightgray')

# Add data rows
for row_idx, row in enumerate(summary_data, start=1):
    for col, text in enumerate(row):
        table.add_cell(row_idx, col, width=1/len(header), height=0.4, text=text, loc='center', edgecolor='black')

ax.add_table(table)

# Save the table as an image
plt.tight_layout()
plt.savefig(r'.\figures\model_summary_table.png')

# %%
