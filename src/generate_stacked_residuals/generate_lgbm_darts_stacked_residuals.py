"""
Created on Wed Sep 25 12:28:04 2024

@author: Jonas Petersen
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
import darts.models as dm
from darts import TimeSeries

import src.generate_stacked_residuals.global_model_parameters as gmp
from src.utils import log_execution_time

model_config_default = {
    'lags': gmp.context_length,
    'n_estimators': 100,
    'learning_rate': 0.01,
    'max_depth': 60,
    'random_state': 42,
    'verbosity': -1,
    # 'lags_past_covariates': gmp.context_length
    }

def generate_lgbm_darts_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        model_config: dict = model_config_default
        ):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_lgbm_darts_stacked_forecast'):
        lgbm_model = dm.LightGBMModel(**model_config)
        lgbm_model.fit(TimeSeries.from_dataframe(df.iloc[:-fh], 'date'))
        model_forecasts = pd.DataFrame(data = lgbm_model.predict(forecast_horizon).values(), columns = time_series_column_group)
        residuals.append((df.iloc[-fh:].head(forecast_horizon)[time_series_column_group]-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle("./data/results/stacked_residuals_lgbm_darts.pkl")

log_execution_time(
    generate_lgbm_darts_stacked_residuals,
    gmp.log_file,
    "generate_lgbm_darts_stacked_residuals"
)