"""
Created on Thu Sep 26 13:16:23 2024

@author: petersen.jonas
"""

from tqdm import tqdm
import pandas as pd
from sorcerer.sorcerer_model import SorcererModel
from src.utils import suppress_output
import src.generate_stacked_residuals.global_model_parameters as gmp
from src.utils import log_execution_time
from joblib import Parallel, delayed

sampler_config_default = {
    "draws": 500,
    "tune": 300,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "verbose": True,
    "nuts_sampler": "numpyro"
}

sampler_config_MAP = {
    "draws": 500,
    "tune": 300,
    "chains": 1,
    "cores": 1,
    "sampler": "MAP",
    "verbose": True,
    "nuts_sampler": "numpyro"
}

model_config_default = {
    "delta_mu_prior": 0,
    "delta_b_prior": 0.1,
    "m_sigma_prior": 0.2,
    "k_sigma_prior": 0.2,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior": 1,
    "precision_target_distribution_prior_alpha": 50,
    "precision_target_distribution_prior_beta": 0.1,
    "single_scale_mu_prior": 0,
    "single_scale_sigma_prior": 1,
    "shared_scale_mu_prior": 1,
    "shared_scale_sigma_prior": 1,
    "number_of_individual_trend_changepoints": None,  # Placeholder value
    "individual_fourier_terms": [
        {'seasonality_period_baseline': gmp.number_of_weeks_in_a_year,'number_of_fourier_components': 10}
    ],
    "shared_fourier_terms": [
        {'seasonality_period_baseline': gmp.number_of_weeks_in_a_year,'number_of_fourier_components': 8},
        {'seasonality_period_baseline': gmp.number_of_weeks_in_a_year/4,'number_of_fourier_components': 4},
        {'seasonality_period_baseline': gmp.number_of_weeks_in_a_year/12,'number_of_fourier_components': 2},
    ]
}

def generate_forecast(fh, df, forecast_horizon, sampler_config, model_config, time_series_column_group):
    training_data = df.iloc[:-fh]
    testing_data = df.iloc[-fh:].head(forecast_horizon)
    y_train_min = training_data[time_series_column_group].min()
    y_train_max = training_data[time_series_column_group].max()
    model_config["number_of_individual_trend_changepoints"] = int(len(training_data) / gmp.number_of_weeks_in_a_year)  # Dynamically update
    sorcerer = SorcererModel(
        model_config = model_config,
        model_name = "SorcererModel",
        model_version = "v0.4.1"
    )
    suppress_output(func=sorcerer.fit, training_data=training_data, sampler_config=sampler_config)
    model_forecasts = suppress_output(sorcerer.point_estimate, test_data = testing_data, point_estimate="median")
    return (testing_data.reset_index(drop = True)-model_forecasts)[time_series_column_group]

def generate_sorcerer_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        sampler_config: dict = sampler_config_default,
        model_config: dict = model_config_default
):
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = Parallel(n_jobs= gmp.n_jobs)(delayed(generate_forecast)(fh, df, forecast_horizon, sampler_config, model_config, time_series_column_group) 
                                     for fh in tqdm(range(forecast_horizon, forecast_horizon + simulated_number_of_forecasts), desc='generate_sorcerer_stacked_residuals'))

    pd.concat(residuals, axis=0).to_pickle(f"./data/results/stacked_residuals_sorcerer_{sampler_config['sampler']}.pkl")

log_execution_time(
    lambda: generate_sorcerer_stacked_residuals(sampler_config=sampler_config_MAP),
    gmp.log_file,
    "generate_sorcerer_MAP_stacked_residuals"
)

log_execution_time(
    lambda: generate_sorcerer_stacked_residuals(),
    gmp.log_file,
    "generate_sorcerer_NUTS_stacked_residuals"
)