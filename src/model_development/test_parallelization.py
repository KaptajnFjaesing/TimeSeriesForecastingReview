
"""
Created on Wed Sep 25 12:16:45 2024

@author: Jonas Petersen
"""

import numpy as np
import pandas as pd
import joblib

from tlp_regression.tlp_regression_model import TlpRegressionModel

from src.utils import suppress_output
import src.generate_stacked_residuals.global_model_parameters as gmp

#%%

sampler_config_default = {
    "draws": 300,
    "tune": 100,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "nuts_sampler": "numpyro",
    "verbose": False
}

sampler_config_MAP = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "MAP",
    "nuts_sampler": "numpyro",
    "verbose": False
}

base_alpha = 2
base_beta = 0.1

model_config = {
    "w_input_alpha_prior": base_alpha,
    "b_input_alpha_prior": base_alpha,
    "w2_alpha_prior": base_alpha,
    "b2_alpha_prior": base_alpha,
    "w_output_alpha_prior": base_alpha,
    "b_output_alpha_prior": base_alpha,
    "precision_alpha_prior": 100,
    "w_input_beta_prior": base_beta,
    "b_input_beta_prior": base_beta,
    "w2_beta_prior": base_beta,
    "b2_beta_prior": base_beta,
    "w_output_beta_prior": base_beta,
    "b_output_beta_prior": base_beta,
    "precision_beta_prior": 0.1,
    "activation": "swish",
    "n_hidden_layer1": 20,
    "n_hidden_layer2": 20,
    'number_of_weeks_in_a_year': gmp.number_of_weeks_in_a_year
}

df = gmp.df

tlp_regression_model = TlpRegressionModel(
    model_config = model_config,
    model_name = "TLPRegressionModel",
    model_version = "v0.1.0"
    )
time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
df_time_series = df.copy(deep = True)
df_time_series.loc[:, 'week'] = np.sin(2*np.pi*df_time_series['date'].dt.strftime('%U').astype(int)/model_config['number_of_weeks_in_a_year'])
df_time_series.loc[:, "month"] = np.sin(2*np.pi*df_time_series["date"].dt.month/12)
df_time_series.loc[:, "quarter"] = np.sin(2*np.pi*df_time_series["date"].dt.quarter/4)

def generate_tlp_regression_model_stacked_residuals(
        fh: int,
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        sampler_config: dict = sampler_config_default
        ):
    training_data = df_time_series.iloc[:-fh].copy(deep=True)
    testing_data = df_time_series.iloc[-fh:].head(forecast_horizon).copy(deep=True)
    training_min_year = training_data["date"].dt.year.min()
    training_max_year = training_data["date"].dt.year.max()
    training_data.loc[:, "year"] = (training_data["date"].dt.year - training_min_year) / (training_max_year - training_min_year)
    testing_data.loc[:, "year"] = (testing_data["date"].dt.year - training_min_year) / (training_max_year - training_min_year)
    y_train_min = training_data[time_series_column_group].min()
    y_train_max = training_data[time_series_column_group].max()
    x_train = training_data[["week", "month", "quarter", "year"]].values
    y_train = ((training_data[time_series_column_group]-y_train_min)/(y_train_max-y_train_min)).values
    x_test = testing_data[["week", "month", "quarter", "year"]].values   
    suppress_output(tlp_regression_model.fit, X = x_train, y = y_train, sampler_config = sampler_config)
    model_preds = suppress_output(tlp_regression_model.sample_posterior_predictive, x_test=x_test)
    model_forecasts_unnormalized = pd.DataFrame(data = model_preds["target_distribution"].mean(("chain", "draw")), columns = time_series_column_group)
    model_forecasts = model_forecasts_unnormalized*(y_train_max-y_train_min)+y_train_min
    return (testing_data[time_series_column_group]-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True)


residuals = joblib.Parallel(n_jobs = 10-1)(joblib.delayed(generate_tlp_regression_model_stacked_residuals)(fh, df, gmp.forecast_horizon, sampler_config_default ) for fh in range(gmp.forecast_horizon,gmp.forecast_horizon+gmp.simulated_number_of_forecasts))
pd.concat(list(residuals), axis=0).to_pickle("./data/results/stacked_residuals_tlp_{sampler_config_default['sampler']}.pkl")

residuals_MAP = joblib.Parallel(n_jobs = 10-1)(joblib.delayed(generate_tlp_regression_model_stacked_residuals)(fh, df, gmp.forecast_horizon, sampler_config_MAP ) for fh in range(gmp.forecast_horizon,gmp.forecast_horizon+gmp.simulated_number_of_forecasts))
pd.concat(list(residuals_MAP), axis=0).to_pickle("./data/results/stacked_residuals_tlp_MAP.pkl")
