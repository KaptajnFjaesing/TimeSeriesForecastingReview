"""
Created on Thu Sep 26 13:16:23 2024

@author: petersen.jonas
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
from tlp_regression.tlp_regression_model import TlpRegressionModel
from src.utils import suppress_output
import src.generate_stacked_residuals.global_model_parameters as gmp
from src.utils import log_execution_time

from joblib import Parallel, delayed


sampler_config_default = {
    "draws": 300,
    "tune": 100,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
#    "nuts_sampler": "numpyro",
    "verbose": False
}

sampler_config_MAP = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "MAP",
#s    "nuts_sampler": "numpyro",
    "verbose": False
}

base_alpha = 2
base_beta = 0.1

model_config_default = {
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


def generate_forecast(fh, df, forecast_horizon, sampler_config, model_config, time_series_column_group):
    training_data = df.iloc[:-fh].copy(deep=True)
    testing_data = df.iloc[-fh:].head(forecast_horizon).copy(deep=True)
    training_min_year = training_data["date"].dt.year.min()
    training_max_year = training_data["date"].dt.year.max()
    training_data.loc[:, "year"] = (training_data["date"].dt.year - training_min_year) / (training_max_year - training_min_year)
    testing_data.loc[:, "year"] = (testing_data["date"].dt.year - training_min_year) / (training_max_year - training_min_year)

    y_train_min = training_data[time_series_column_group].min()
    y_train_max = training_data[time_series_column_group].max()

    x_train = training_data[["week", "month", "quarter", "year"]].values
    y_train = ((training_data[time_series_column_group] - y_train_min) / (y_train_max - y_train_min)).values
    x_test = testing_data[["week", "month", "quarter", "year"]].values

    tlp_regression_model = TlpRegressionModel(
        model_config=model_config,
        model_name="TLPRegressionModel",
        model_version="v0.1.0"
    )

    suppress_output(tlp_regression_model.fit, X=x_train, y=y_train, sampler_config=sampler_config)
    model_preds = suppress_output(tlp_regression_model.sample_posterior_predictive, x_test=x_test)

    model_forecasts_unnormalized = pd.DataFrame(data=model_preds["target_distribution"].median(("chain", "draw")), columns=time_series_column_group)
    model_forecasts = model_forecasts_unnormalized * (y_train_max - y_train_min) + y_train_min

    return (testing_data[time_series_column_group] - model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop=True)


def generate_tlp_regression_model_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        sampler_config: dict = sampler_config_default,
        model_config: dict = model_config_default
        ) -> None:
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    df_time_series = df.copy(deep=True)
    df_time_series.loc[:, 'week'] = np.sin(2*np.pi*df_time_series['date'].dt.strftime('%U').astype(int) / model_config['number_of_weeks_in_a_year'])
    df_time_series.loc[:, "month"] = np.sin(2*np.pi*df_time_series["date"].dt.month / 12)
    df_time_series.loc[:, "quarter"] = np.sin(2*np.pi*df_time_series["date"].dt.quarter / 4)

    residuals = Parallel(n_jobs=gmp.n_jobs)(delayed(generate_forecast)(
        fh, df_time_series, forecast_horizon, sampler_config, model_config, time_series_column_group
    ) for fh in tqdm(range(forecast_horizon, forecast_horizon + simulated_number_of_forecasts), desc='generate_tlp_regression_model_stacked_residuals'))

    pd.concat(residuals, axis=0).to_pickle(f"./data/results/stacked_residuals_tlp_{sampler_config['sampler']}.pkl")

log_execution_time(
    lambda: generate_tlp_regression_model_stacked_residuals(sampler_config=sampler_config_MAP),
    gmp.log_file,
    "generate_tlp_MAP_regression_model_stacked_residuals"
)

log_execution_time(
    lambda: generate_tlp_regression_model_stacked_residuals(),
    gmp.log_file,
    "generate_tlp_NUTS_regression_model_stacked_residuals"
)