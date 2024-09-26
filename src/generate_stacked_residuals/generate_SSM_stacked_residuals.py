"""
Created on Wed Sep 25 10:22:56 2024

@author: Jonas Petersen
"""

from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm

from src.utils import suppress_output
import src.generate_stacked_residuals.global_model_parameters as gmp


model_config_default = {
    'freq_seasonal': [{
        'period': int(gmp.number_of_weeks_in_a_year),
        'harmonics':15
        }],
    'autoregressive': 5,
    'level': True,
    'trend': True,
    'stochastic_level': True, 
    'stochastic_trend': True,
    'irregular': True
    }

def generate_SSM_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        model_config: dict = model_config_default
        ):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_SSM_stacked_residuals'):
        training_data = df.iloc[:-fh]
        y_train_min = training_data[time_series_column_group].min()
        y_train_max = training_data[time_series_column_group].max()
        y_train = (training_data[time_series_column_group]-y_train_min)/(y_train_max-y_train_min)
        model_forecasts = pd.DataFrame(columns = time_series_column_group)
        for time_series_column in time_series_column_group:
            struobs = sm.tsa.UnobservedComponents(y_train[time_series_column],**model_config)
            mlefit = suppress_output(struobs.fit, disp = 0)
            model_results = mlefit.get_forecast(forecast_horizon).summary_frame(alpha=0.05)
            model_forecasts[time_series_column] = model_results['mean']*(y_train_max[time_series_column]-y_train_min[time_series_column])+y_train_min[time_series_column]
        test_data = df[time_series_column_group].iloc[-fh:].head(forecast_horizon)
        residuals.append((test_data-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle('./data/results/stacked_residuals_statespace.pkl')

generate_SSM_stacked_residuals()