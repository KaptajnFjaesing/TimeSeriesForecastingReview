"""
Created on Wed Sep 25 10:56:33 2024

@author: Jonas Petersen
"""

from tqdm import tqdm
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.utils import suppress_output
import src.generate_stacked_residuals.global_model_parameters as gmp
from src.utils import log_execution_time

model_config_default = {
    'seasonal': 'add',
    'trend': 'add',
    'seasonal_periods': int(gmp.number_of_weeks_in_a_year),
    'freq': "W-MON"
    }

def generate_exponential_smoothing_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        model_config: dict = model_config_default
        ):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_exponential_smoothing_stacked_residuals'):
        training_data = df.iloc[:-fh].set_index('date', inplace=False)
        y_train_min = training_data[time_series_column_group].min()
        y_train_max = training_data[time_series_column_group].max()
        y_train = (training_data[time_series_column_group]-y_train_min)/(y_train_max-y_train_min)
        model_forecasts = pd.DataFrame(columns = time_series_column_group)
        for time_series_column in time_series_column_group:
            model = ExponentialSmoothing(y_train[time_series_column], **model_config)
            fit_model = suppress_output(model.fit)
            model_results = fit_model.forecast(steps=forecast_horizon)
            model_forecasts[time_series_column] = (model_results*(y_train_max[time_series_column]-y_train_min[time_series_column])+y_train_min[time_series_column])
        test_data = df[time_series_column_group].iloc[-fh:].head(forecast_horizon)
        residuals.append((test_data-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle('./data/results/stacked_residuals_exponential_smoothing.pkl')

log_execution_time(
    generate_exponential_smoothing_stacked_residuals,
    gmp.log_file,
    "generate_exponential_smoothing_stacked_residuals"
)