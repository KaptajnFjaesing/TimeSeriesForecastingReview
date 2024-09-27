"""
Created on Thu Sep 26 21:08:29 2024

@author: Jonas Petersen
"""

from tqdm import tqdm
import pandas as pd
import numpy as np

from src.utils import CustomBackTransformation
import src.generate_stacked_residuals.global_model_parameters as gmp

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
    df_temp = df.copy(deep = True)
    df_temp.set_index('date', inplace=True, drop = True)
    full_index = pd.date_range(start=df_temp.index.min(), end=df_temp.index.max(), freq='W-MON')
    df_temp.index = full_index
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    df_temp[time_series_column_group] = np.log(df_temp[time_series_column_group])
    df_0 = df_temp[time_series_column_group].iloc[0,:] # save first datapoint
    df_temp = df_temp.diff().bfill()
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_deepar_stacked_residuals'):
        training_data = df_temp.iloc[:-fh*2]
        validation_data = df_temp.iloc[:-fh]
        df_1 = np.sum(validation_data)
        y_train_melt = pd.melt(training_data,value_vars=training_data.columns,var_name='ts',value_name='target',ignore_index=False)
        y_validation_melt = pd.melt(validation_data,value_vars=training_data.columns,var_name='ts',value_name='target',ignore_index=False)
        dataset = PandasDataset.from_long_dataframe(y_train_melt,target='target',item_id='ts')
        dataset_validation = PandasDataset.from_long_dataframe(y_validation_melt,target='target',item_id='ts')
 
        gtmodel = gt.DeepAREstimator(**model_config_default).train(
            training_data=dataset,
            validation_data=dataset_validation,
            verbose = False
            )
        
        backtransformation = CustomBackTransformation(constants0=df_0.values,consants1=df_1.values)
        forecasts = list(gtmodel.predict(dataset_validation))
        transformed_forecasts = [backtransformation(forecast, i) for i, forecast in enumerate(forecasts)]
        model_forecasts = pd.DataFrame(columns = time_series_column_group)
        for j in range(len(time_series_column_group)):
            model_forecasts[time_series_column_group[j]] = transformed_forecasts[j].samples.mean(axis=0)
        residuals.append(df.iloc[-fh:].head(forecast_horizon)[time_series_column_group].reset_index(drop = True)-model_forecasts.reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle("./data/results/stacked_residuals_deepar.pkl")

generate_deepar_stacked_residuals()
