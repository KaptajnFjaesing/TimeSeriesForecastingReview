"""
Created on Wed Sep 25 12:32:06 2024

@author: Jonas Petersen

TiDEModel paper:
https://arxiv.org/pdf/2304.08424
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import darts.models as dm
from darts import TimeSeries
from joblib import Parallel, delayed
import logging
import torch

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from src.utils import suppress_output
import src.generate_stacked_residuals.global_model_parameters as gmp


model_config_default = {
    'input_chunk_length': 64,
    'hidden_size': 256,
    'n_epochs': 20,
    'dropout': 0.3,
    'optimizer_cls': torch.optim.AdamW,
    'optimizer_kwargs': {'lr': 0.001, 'weight_decay': 1e-12, 'amsgrad': True},
    'random_state': 42,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'decoder_output_dim': 8,
    'temporal_decoder_hidden': 32,
    'use_layer_norm': True,
    'batch_size': 5,
    }

"""
NOTE:
According to the Darts documentation, their deep learning models should have a parameter, where the number of CPU workers can be set.
However, this does not seem to work. You should also be able to pass a parameter to the underlying PyTorch dataloader, but this also does not seem to work.
"""
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

def generate_forecast(fh, df, df_features, forecast_horizon, model_config, time_series_column_group, target_columns, feature_columns):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    x_train = df_features[feature_columns + ['date']].iloc[:-fh]
    y_train = df_features[target_columns + ['date']].iloc[:-fh]
    x_test = df_features[feature_columns + ['date']].iloc[-fh:].head(forecast_horizon)
    baseline = df_features[time_series_column_group].loc[x_test.index[0] - 1]
    tide_model = dm.TiDEModel(**model_config, output_chunk_length = forecast_horizon)
    # future_covariates = TimeSeries.from_dataframe(pd.concat([x_train[['date', 'time_sine']], x_test[['date', 'time_sine']]]), 'date')
    # past_covariates = TimeSeries.from_dataframe(x_train[[x for x in x_train.columns if x != 'time_sine']], 'date')
    suppress_output(tide_model.fit, TimeSeries.from_dataframe(y_train, 'date'))
    predictions = suppress_output(tide_model.predict, forecast_horizon).values()
    data = np.einsum("i, ji -> ji", baseline, np.exp(np.cumsum(predictions, axis=0)))
    model_forecasts = pd.DataFrame(data = data, columns = time_series_column_group)
    residual = (df.iloc[-fh:].head(forecast_horizon)[time_series_column_group] - model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True)
    return residual

def generate_tide_feature_darts_stacked_residuals(
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
    residuals = Parallel(n_jobs=2)(delayed(generate_forecast)(
        fh, df, df_features, forecast_horizon, model_config, time_series_column_group, target_columns, feature_columns
    ) for fh in tqdm(range(forecast_horizon, forecast_horizon + simulated_number_of_forecasts), desc='generate_SSM_stacked_residuals'))
    pd.concat(residuals, axis=0).to_pickle("./data/results/stacked_residuals_tide_feature_darts.pkl")

generate_tide_feature_darts_stacked_residuals()