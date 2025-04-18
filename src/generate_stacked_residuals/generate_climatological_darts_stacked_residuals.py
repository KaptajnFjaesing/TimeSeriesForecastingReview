"""
Created on Wed Sep 25 12:32:06 2024

@author: Jonas Petersen

TiDEModel paper:
https://arxiv.org/pdf/2304.08424
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel

from src.utils import suppress_output
from src.utils import log_execution_time
import src.generate_stacked_residuals.global_model_parameters as gmp


class Climatological(ForecastingModel):
    def __init__(self):
        super().__init__(add_encoders=0)
        
    def _is_probabilistic(self):
        return True
    
    def fit(self, series: TimeSeries):
        super().fit(series)
    
    def predict(self, n: int, num_samples: int = 1):
        values = self.training_series.values().squeeze()
        samples = np.random.choice(values[np.isfinite(values)], (n, num_samples))
        return self._build_forecast_series(np.expand_dims(samples, axis=1), self.training_series)
    
    def _model_encoder_settings(self):
        pass
    
    def extreme_lags(self):
        pass

    def supports_multivariate(self):
        return False

    def supports_transferrable_series_prediction(self):
        pass

def generate_climatological_darts_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        ):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_climatological_darts_stacked_residuals'):
        model_forecasts = pd.DataFrame(columns = time_series_column_group)
        for time_series_column in time_series_column_group:
            model = Climatological()
            suppress_output(model.fit, TimeSeries.from_dataframe(df.iloc[:-fh][['date', time_series_column]], 'date'))
            predictions = model.predict(n = forecast_horizon, num_samples = 100)
            model_forecasts[time_series_column] = predictions.values().squeeze()
        residuals.append((df.iloc[-fh:].head(forecast_horizon)[time_series_column_group]-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle("./data/results/stacked_residuals_climatological_darts.pkl")
    print("generate_climatological_darts_stacked_residuals completed")

log_execution_time(
    generate_climatological_darts_stacked_residuals,
    gmp.log_file,
    "generate_climatological_darts_stacked_residuals"
)