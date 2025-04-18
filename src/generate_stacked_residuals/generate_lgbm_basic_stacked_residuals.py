"""
Created on Wed Sep 25 11:46:29 2024

@author: Jonas Petersen
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
import lightgbm as lgbm

import src.generate_stacked_residuals.global_model_parameters as gmp
from src.utils import log_execution_time

model_config_default = {
    'verbose': -1,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 5,
    'learning_rate': 0.1,
    'feature_fraction': 0.9
}

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

def light_gbm_forecast(
        x_train:pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        time_series_column: str,
        forecast_horizon: int,
        context_length: int,
        seasonality_period: float,
        model_config: dict
        ):
    
    train_dataset = lgbm.Dataset(
        data = x_train.values,
        label = y_train.values
        )
    model = lgbm.train(
        params = model_config,
        train_set = train_dataset, 
        num_boost_round = 2000
        )

    predictions = light_gbm_predict(
        x_test = x_test,
        forecast_horizon = forecast_horizon,
        context_length = context_length,
        seasonality_period = seasonality_period,
        model = model
        )
    return predictions


def generate_lgbm_basic_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        forecast_horizon: int = gmp.forecast_horizon,
        context_length: int = gmp.context_length,
        seasonality_period: float = gmp.number_of_weeks_in_a_year,
        model_config: dict = model_config_default
        ):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    feature_columns = [str(x) for x in range(1,context_length)]+['time_sine']
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_lgbm_basic_stacked_residuals'):
        model_forecasts = pd.DataFrame(columns = time_series_column_group)
        for time_series_column in time_series_column_group:
            df_0, df_features = feature_generation(
                df = df,
                time_series_column = time_series_column,
                context_length = context_length,
                seasonality_period = seasonality_period
                )
            x_train = df_features[feature_columns].iloc[:-fh]
            y_train = df_features[time_series_column+'_log_diff'].iloc[:-fh]
            x_test = df_features[feature_columns].iloc[-fh:].head(forecast_horizon)
            baseline = df_features[time_series_column][x_test.index-1].iloc[0]
            model_results = light_gbm_forecast(
                    x_train = x_train,
                    y_train = y_train,
                    x_test = x_test,
                    time_series_column = time_series_column,
                    forecast_horizon = forecast_horizon,
                    context_length = context_length,
                    seasonality_period = seasonality_period,
                    model_config = model_config
                    )
            model_forecasts[time_series_column] = baseline*np.exp(np.cumsum(model_results))
        test_data = df[time_series_column_group].iloc[-fh:].head(forecast_horizon)
        residuals.append((test_data-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle('./data/results/stacked_residuals_lgbm_basic.pkl')

log_execution_time(
    generate_lgbm_basic_stacked_residuals,
    gmp.log_file,
    "generate_lgbm_basic_stacked_residuals"
)