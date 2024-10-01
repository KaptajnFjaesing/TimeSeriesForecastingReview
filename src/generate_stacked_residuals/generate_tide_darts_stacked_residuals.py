"""
Created on Wed Sep 25 12:32:06 2024

@author: Jonas Petersen

TiDEModel paper:
https://arxiv.org/pdf/2304.08424
"""
from tqdm import tqdm
import pandas as pd
import darts.models as dm
from darts import TimeSeries
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
def generate_tide_darts_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        model_config: dict = model_config_default
        ):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_tide_darts_stacked_residuals'):
        tide_model = dm.TiDEModel(**model_config, output_chunk_length = forecast_horizon)
        suppress_output(tide_model.fit, TimeSeries.from_dataframe(df.iloc[:-fh], 'date'))
        predictions = suppress_output(tide_model.predict, forecast_horizon)
        model_forecasts = pd.DataFrame(data = predictions.values(), columns = time_series_column_group)
        residuals.append((df.iloc[-fh:].head(forecast_horizon)[time_series_column_group]-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle("./data/results/stacked_residuals_tide_darts.pkl")

generate_tide_darts_stacked_residuals()