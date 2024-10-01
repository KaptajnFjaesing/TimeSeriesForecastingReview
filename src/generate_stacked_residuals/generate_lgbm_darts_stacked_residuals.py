"""
Created on Wed Sep 25 12:28:04 2024

@author: Jonas Petersen
"""
# %%
from tqdm import tqdm
import pandas as pd
import numpy as np
import darts.models as dm
from darts import TimeSeries

import src.generate_stacked_residuals.global_model_parameters as gmp

model_config_default = {
    'lags': gmp.context_length,
    'n_estimators': 100,
    'learning_rate': 0.01,
    'max_depth': 60,
    'random_state': 42,
    'verbosity': -1,
    # 'lags_past_covariates': gmp.context_length
    }

def feature_generation(
        df: pd.DataFrame,
        time_series_column_group: list[str],
        context_length: int,
        seasonality_period: float):
    dict_features = df.to_dict("list")
    for time_series_column in time_series_column_group:
        dict_features[time_series_column+'_log'] = np.log(dict_features[time_series_column]) # Log-scale
        dict_features[time_series_column+'_log_diff'] = np.insert(np.diff(dict_features[time_series_column+'_log']), 0, np.nan) # gradient
        for i in range(1,context_length):
            # Fill lagged values as features
            dict_features[f'{time_series_column}_{i}'] = np.zeros(len(dict_features[time_series_column+'_log_diff']))
            dict_features[f'{time_series_column}_{i}'][:i] = np.nan
            dict_features[f'{time_series_column}_{i}'][i:] = dict_features[time_series_column+'_log_diff'][:-i]
    df_features = pd.DataFrame(dict_features)
    df_features.dropna(inplace=True)
    df_features['time_sine'] = np.sin(2*np.pi*df_features.index/seasonality_period) # Create a sine time-feature with the period set to 12 months
    
    return df_features

def generate_lgbm_darts_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        model_config: dict = model_config_default
        ):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    df_features = feature_generation(df, time_series_column_group, gmp.context_length, gmp.number_of_weeks_in_a_year)
    residuals = []
    target_columns = [x +'_log_diff' for x in time_series_column_group]
    feature_columns = [f'{x}_{i}' for x in time_series_column_group for i in range(1, gmp.context_length)]+['time_sine']
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_lgbm_darts_stacked_forecast'):
        x_train = df_features[feature_columns + ['date']].iloc[:-fh]
        y_train = df_features[target_columns + ['date']].iloc[:-fh]
        x_test = df_features[feature_columns + ['date']].iloc[-fh:].head(forecast_horizon)
        baseline = df_features[time_series_column_group].loc[x_test.index[0] - 1]
        lgbm_model = dm.LightGBMModel(**model_config)
        lgbm_model.fit(series = TimeSeries.from_dataframe(y_train, 'date'))
        predictions = lgbm_model.predict(forecast_horizon).values()
        data = np.einsum("i, ji -> ji", baseline, np.exp(np.cumsum(predictions, axis=0)))
        model_forecasts = pd.DataFrame(data = data, columns = time_series_column_group)
        residuals.append((df.iloc[-fh:].head(forecast_horizon)[time_series_column_group] - model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle("./data/results/stacked_residuals_lgbm_darts.pkl")

generate_lgbm_darts_stacked_residuals()