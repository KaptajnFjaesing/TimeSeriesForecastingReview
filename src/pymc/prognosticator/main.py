"""
Created on Mon Sep 16 08:37:30 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
from examples.load_data import normalized_weekly_store_category_household_sales


df = normalized_weekly_store_category_household_sales()


def create_lagged_features(df, time_series_columns, context_length, forecast_horizon):
    """Create lagged features for a given column in the DataFrame, with future targets."""
    
    for column_name in time_series_columns:
        lagged_df = pd.DataFrame()
        
        # Create lagged features
        for i in range(context_length, -1, -1):
            lagged_df[f'lag_{i}'] = df[column_name].shift(i)
        
        # Create future targets
        for i in range(1, forecast_horizon + 1):
            lagged_df[f'target_{i}'] = df[column_name].shift(-i)
    
    lagged_df.dropna(inplace=True)  # Remove rows with NaN values due to shifting
    return lagged_df

# Parameters
n_lags = 35  # Number of lagged time steps to use as features
forecast_horizon = 26  # Number of future time steps to predict

# Create lagged features and future targets
lagged_features = create_lagged_features(df, 'Passengers', n_lags = n_lags, n_ahead = forecast_horizon)

# Split the data
train_size = int(len(lagged_features) * 0.6)  # 60% for training, 40% for testing
training_data = lagged_features.iloc[:train_size]
test_data = lagged_features.iloc[train_size:]

normalization = 400
# Define features and targets
x_train = training_data[training_data.columns[:n_lags+1]]/normalization  # Lagged features
y_train = training_data[training_data.columns[n_lags+1:]]/normalization  # Future targets

x_test = test_data[test_data.columns[:n_lags+1]]/normalization  # Lagged features
y_test = test_data[test_data.columns[n_lags+1:]]/normalization  # Future targets

if len(x_test)<forecast_horizon:    
    print("WARNING: Forecast_horizon too long relative to test/training split")

# %% Define model

time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

df_time_series = df[time_series_columns]

model_name = "PrognosticatorModel"
version = "v0.1"
method = "MAP"



forecast_horizon = 52

training_data = df_time_series.iloc[:-forecast_horizon]
test_data = df_time_series.iloc[-forecast_horizon:]

y_train_min = training_data[unnormalized_column_group].min()
y_train_max = training_data[unnormalized_column_group].max()

# Sorcerer
sampler_config = {
    "draws": 2000,
    "tune": 500,
    "chains": 1,
    "cores": 1
}

model_config = {
    "test_train_split": len(training_data)/len(df_time_series),
    "number_of_individual_trend_changepoints": 10,
    "number_of_individual_fourier_components": 5,
    "number_of_shared_fourier_components": 3,
    "number_of_shared_seasonality_groups": 1,
    "delta_mu_prior": 0,
    "delta_b_prior": 0.3,
    "m_sigma_prior": 3,
    "k_sigma_prior": 3,
    "precision_target_distribution_prior_alpha": 100,
    "precision_target_distribution_prior_beta": 0.05,
    "relative_uncertainty_factor_prior": 1000
}

if method == "MAP":
    model_config['precision_target_distribution_prior_alpha'] = 100
    model_config['precision_target_distribution_prior_beta'] = 0.05

sorcerer = SorcererModel(
    model_config = model_config,
    model_name = model_name,
    version = version
    )


# %% Fit model

sorcerer.fit(
    training_data = training_data,
    seasonality_periods = seasonality_periods,
    method = method
    )

if method != "MAP":
    fname = "examples/models/sorcer_model_v02.nc"
    sorcerer.save(fname)

#%% Produce forecast

"""
Load from stored model
fname = "examples/models/sorcer_model_v01.nc"
sorcerer.load(fname)
"""

(preds_out_of_sample, model_preds) = sorcerer.sample_posterior_predictive(test_data = df_time_series)

#%% Plot forecast along with test data
(X_train, y_train, X_test, y_test) = sorcerer.normalize_data(
        training_data,
        test_data
        )

hdi_values = az.hdi(model_preds)["target_distribution"].transpose("hdi", ...)

# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil((len(time_series_columns)-1) / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows   
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(len(time_series_columns)-1):
    ax = axs[i]  # Get the correct subplot
    
    ax.plot(X_train, y_train[y_train.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(X_test, y_test[y_test.columns[i]], color = 'black',  label='Test Data')
    ax.plot(preds_out_of_sample, (model_preds["target_distribution"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    ax.fill_between(
        preds_out_of_sample.values,
        hdi_values[0].values[:,i],  # lower bound of the HDI
        hdi_values[1].values[:,i],  # upper bound of the HDI
        color= 'blue',   # color of the shaded region
        alpha=0.4,      # transparency level of the shaded region
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure

# %%
plt.savefig('./examples/figures/forecast.png')
