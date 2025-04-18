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

from src.utils import suppress_output
import src.generate_stacked_residuals.global_model_parameters as gmp
from src.utils import log_execution_time

model_config_default = {
    'K': int(gmp.number_of_weeks_in_a_year),
    }

def generate_naive_seasonal_darts_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        model_config: dict = model_config_default
        ):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_naive_seasonal_darts_stacked_residuals'):
        model = dm.NaiveSeasonal(**model_config)
        suppress_output(model.fit, TimeSeries.from_dataframe(df.iloc[:-fh], 'date'))
        predictions = suppress_output(model.predict, forecast_horizon)
        model_forecasts = pd.DataFrame(data = predictions.values(), columns = time_series_column_group)
        residuals.append((df.iloc[-fh:].head(forecast_horizon)[time_series_column_group]-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle("./data/results/stacked_residuals_naive_seasonal_darts.pkl")

log_execution_time(
    generate_naive_seasonal_darts_stacked_residuals,
    gmp.log_file,
    "generate_naive_seasonal_darts_stacked_residuals"
)