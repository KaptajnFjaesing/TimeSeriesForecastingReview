# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:53:58 2024

@author: petersen.jonas
"""

from src.load_data import normalized_weekly_store_category_household_sales
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

df = normalized_weekly_store_category_household_sales()

# %%

def feature_generation(
        df: pd.DataFrame,
        time_series_column: str,
        context_length: int,
        seasonality_period: float):
    df_features = df.copy()
    df_features[time_series_column+'_log'] = np.log(df_features[time_series_column]) # Log-scale
    df_0 = df_features.iloc[0,:] # save first datapoint
    df_features[time_series_column+'_log_diff'] = df_features[time_series_column+'_log'].diff() # gradient
    for i in range(1,context_length):
        df_features[f'{i}'] = df_features[time_series_column+'_log_diff'].shift(i) # Fill lagged values as features    
    df_features.dropna(inplace=True)
    df_features['time_sine'] = np.sin(2*np.pi*df_features.index/seasonality_period) # Create a sine time-feature with the period set to 12 months
    
    return df_0, df_features

def light_gbm_predict(
        x_test,
        forecast_horizon,
        context_length,
        seasonality_period,
        model
        ):

    predictions = np.zeros(forecast_horizon)
    context = x_test.iloc[0].values.reshape(1,-1)
    test_start_index_plus_one = x_test.index[0]+1

    for i in range(forecast_horizon):
        predictions[i] = model.predict(context)[0]
        context[0,1:context_length] = context[0,0:context_length-1]
        context[0,0] = predictions[i]
        context[0,-1] = np.sin(2*np.pi*(test_start_index_plus_one+i)/seasonality_period) # this is for next iteration
    return predictions


seasonality_period = 52
context_length = 30
forecast_horizon = 52

feature_columns = [str(x) for x in range(1,context_length)]+['time_sine']
time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' in x]

rescaled_forecasts = np.ones((len(time_series_columns),forecast_horizon))

for i in range(len(time_series_columns)):

    df_0, df_features = feature_generation(
        df = df,
        time_series_column = time_series_columns[i],
        context_length = context_length,
        seasonality_period = seasonality_period
        )
    
    x_train = df_features[feature_columns].iloc[:-forecast_horizon]
    y_train = df_features[time_series_columns[i]+'_log_diff'].iloc[:-forecast_horizon]
    
    x_test = df_features[feature_columns].iloc[-forecast_horizon:]
    y_test = df_features[time_series_columns[i]+'_log_diff'].iloc[-forecast_horizon:]

    # Create LightGBM datasets
    train_dataset = lgb.Dataset(
        data = x_train.values,
        label = y_train.values
        )
        
    # Define model parameters
    params = {
        'verbose': -1,
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 5,
        'learning_rate': 0.1,
        'feature_fraction': 0.9
    }
        
    # Train the model with early stopping
    model = lgb.train(
        params,
        train_dataset, 
        num_boost_round = 2000
        )
    
    train_predict = model.predict(x_train.values)
    
    predictions = light_gbm_predict(
        x_test = x_test,
        forecast_horizon = forecast_horizon,
        context_length = context_length,
        seasonality_period = seasonality_period,
        model = model
        )

    baseline = df_features[time_series_columns[i]][x_test.index-1].iloc[0]
    rescaled_forecasts[i] = baseline*np.exp(np.cumsum(predictions))
        

# %%


# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil(len(time_series_columns) / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(len(time_series_columns)):
    ax = axs[i]  # Get the correct subplot
    ax.plot(df['date'], df[time_series_columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(df_features['date'][x_test.index].iloc[-forecast_horizon::], rescaled_forecasts[i], color = 'tab:blue', label='Model')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure




