"""
Created on Thu Sep 26 21:08:29 2024

@author: Jonas Petersen
"""
#%%
from tqdm import tqdm
import pandas as pd
import numpy as np

from src.utils import CustomBackTransformation
import src.generate_stacked_residuals.global_model_parameters as gmp
from src.utils import log_execution_time

import gluonts.torch as gt
from gluonts.dataset.pandas import PandasDataset

import warnings
import logging

# Set the logging level to ERROR to suppress all other messages
logging.getLogger("gluonts").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

model_config_default = {
    'prediction_length': gmp.forecast_horizon,
    'context_length': gmp.context_length,
    'freq': 'W-MON',
    'lr': 0.001, 
    'dropout_rate': 0.2,
    'hidden_size': 60,
    'num_layers': 2,
    'trainer_kwargs': {
        "max_epochs": 10,
        "logger": False,  # Disable logging
        "enable_progress_bar": False,  # Disable progress bar
        "deterministic": True,  # Ensure reproducibility
        "enable_model_summary": False,  # Disable model summary printing
        "fast_dev_run": False,  # Disable fast dev run
        "detect_anomaly": False  # Disable anomaly detection
    }
}

def generate_deepar_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        model_config: dict = model_config_default
        ):
    df_temp = df.copy(deep=True)
    df_temp.set_index('date', inplace=True, drop=True)
    full_index = pd.date_range(start=df_temp.index.min(), end=df_temp.index.max(), freq='W-MON')
    df_temp.index = full_index
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    df_temp[time_series_column_group] = np.log(df_temp[time_series_column_group])
    df_0 = df_temp[time_series_column_group].iloc[0, :]  # Save first datapoint
    df_temp = df_temp.diff().bfill()
    residuals = []
    for fh in tqdm(range(forecast_horizon, forecast_horizon + simulated_number_of_forecasts), desc='generate_deepar_stacked_residuals'):
        training_data = df_temp.iloc[:-2*fh]
        y_train_melt = pd.melt(training_data, value_vars=training_data.columns, var_name='ts', value_name='target', ignore_index=False)
        training_dataset = PandasDataset.from_long_dataframe(y_train_melt, target='target', item_id='ts')

        validation_data = df_temp.iloc[-2*fh:-fh]
        y_validation_melt = pd.melt(validation_data, value_vars=validation_data.columns, var_name='ts', value_name='target', ignore_index=False)
        validation_dataset = PandasDataset.from_long_dataframe(y_validation_melt, target='target', item_id='ts')

        test_data = df_temp.iloc[-fh:]
        y_test_melt = pd.melt(test_data, value_vars=test_data.columns, var_name='ts', value_name='target', ignore_index=False)
        test_dataset = PandasDataset.from_long_dataframe(y_test_melt, target='target', item_id='ts')

        gtmodel = gt.DeepAREstimator(**model_config_default).train(
            training_data=training_dataset,
            validation_data=validation_dataset,
            verbose=False
        )
        forecasts = list(gtmodel.predict(test_dataset))

        model_forecasts = pd.DataFrame(columns=time_series_column_group)
        for idx in range(len(time_series_column_group)):
            model_forecasts[time_series_column_group[idx]] = np.exp(np.cumsum(forecasts[idx].samples, axis=1).mean(axis = 0) + df_0.values[idx] + np.sum(df_temp.iloc[1:-fh]).values[idx])
            #targets = np.exp(np.cumsum(test_data[test_data.columns[idx]]) + df_0.values[idx] + np.sum(df_temp.iloc[1:-fh]).values[idx])
        residuals.append(gmp.df.iloc[-fh:].head(gmp.forecast_horizon)[time_series_column_group].reset_index(drop=True) - model_forecasts.reset_index(drop=True))
    pd.concat(residuals, axis=0).to_pickle("./data/results/stacked_residuals_deepar.pkl")

log_execution_time(
    generate_deepar_stacked_residuals,
    gmp.log_file,
    "generate_deepar_stacked_residuals"
)