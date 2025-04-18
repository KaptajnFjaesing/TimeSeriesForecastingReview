"""
Created on Wed Sep 25 12:21:30 2024

@author: Jonas Petersen
"""

from tqdm import tqdm
import pandas as pd
import darts.models as dm
from darts import TimeSeries

import src.generate_stacked_residuals.global_model_parameters as gmp
from src.utils import log_execution_time

def generate_naive_darts_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts
        ):
    naive_model = dm.NaiveDrift()
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_naive_darts_stacked_residuals'):
        naive_model.fit(TimeSeries.from_dataframe(df.iloc[:-fh], 'date'))
        model_forecasts = pd.DataFrame(data = naive_model.predict(forecast_horizon).values(), columns = time_series_column_group)
        residuals.append((df.iloc[-fh:].head(forecast_horizon)[time_series_column_group]-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle("./data/results/stacked_residuals_naive_darts.pkl")

log_execution_time(
    generate_naive_darts_stacked_residuals,
    gmp.log_file,
    "generate_naive_darts_stacked_residuals"
)