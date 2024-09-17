
"""
Created on Tue Sep 17 12:32:22 2024

@author: petersen.jonas
"""

import numpy as np
import pandas as pd
from src.utils import normalized_weekly_store_category_household_sales
import lightgbm as lgbm

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
# %% Define model
from tqdm import tqdm

time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

df_time_series = df[time_series_columns]

seasonality_period = 52
min_forecast_horizon = 26
max_forecast_horizon = 52
context_length = 30

feature_columns = [str(x) for x in range(1,context_length)]+['time_sine']

model_forecasts_lgbm = []
for forecast_horizon in tqdm(range(min_forecast_horizon,max_forecast_horizon+1)):
    training_data = df_time_series.iloc[:-forecast_horizon]
    test_data = df_time_series.iloc[-forecast_horizon:]
    
    rescaled_forecasts = np.ones((len(unnormalized_column_group),forecast_horizon))
    model_denormalized = []
    for i in range(len(unnormalized_column_group)):
    
        df_0, df_features = feature_generation(
            df = df,
            time_series_column = unnormalized_column_group[i],
            context_length = context_length,
            seasonality_period = seasonality_period
            )
        
        x_train = df_features[feature_columns].iloc[:-forecast_horizon]
        y_train = df_features[unnormalized_column_group[i]+'_log_diff'].iloc[:-forecast_horizon]
        
        x_test = df_features[feature_columns].iloc[-forecast_horizon:]
        y_test = df_features[unnormalized_column_group[i]+'_log_diff'].iloc[-forecast_horizon:]
        
        # Create LightGBM datasets
        train_dataset = lgbm.Dataset(
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
        model = lgbm.train(
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
        
        baseline = df_features[unnormalized_column_group[i]][x_test.index-1].iloc[0]    
        model_denormalized.append(pd.Series(baseline*np.exp(np.cumsum(predictions))))
    model_forecasts_lgbm.append(model_denormalized)



# %%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_test.index,baseline*np.exp(np.cumsum(y_test.values)))
plt.plot(x_test.index,baseline*np.exp(np.cumsum(predictions)))
plt.plot(df_features[unnormalized_column_group[i]], alpha = 0.5)

#%% Compute MASEs
from src.utils import compute_residuals

abs_mean_gradient_training_data = pd.read_pickle('./data/results/abs_mean_gradient_training_data.pkl')

stacked_lgbm = compute_residuals(
         model_forecasts = model_forecasts_lgbm,
         test_data = test_data[unnormalized_column_group],
         min_forecast_horizon = min_forecast_horizon
         )

average_abs_residual_lgbm = stacked_lgbm.abs().groupby(stacked_lgbm.index).mean() # averaged over rolls
average_abs_residual_lgbm.columns = unnormalized_column_group
MASE_lgbm = average_abs_residual_lgbm/abs_mean_gradient_training_data


#%%
stacked_lgbm.to_pickle(r'.\data\results\stacked_forecasts_light_gbm.pkl')