"""
Created on Wed Sep 25 11:19:17 2024

@author: Jonas Petersen
"""

from tqdm import tqdm
import pandas as pd
from sorcerer.sorcerer_model import SorcererModel

from src.utils import suppress_output
import src.generate_stacked_residuals.global_model_parameters as gmp


sampler_config_default = {
    "draws": 500,
    "tune": 100,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "discard_tuned_samples": True,
    "verbose": False
}

sampler_config_MAP = {
    "draws": 500,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "MAP",
    "discard_tuned_samples": True,
    "verbose": False
}

model_config_default = {
    "number_of_individual_trend_changepoints": 40,
    "delta_mu_prior": 0,
    "delta_b_prior": 0.1,
    "m_sigma_prior": 0.2,
    "k_sigma_prior": 0.2,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior" : 1,
    "precision_target_distribution_prior_alpha": 1000,
    "precision_target_distribution_prior_beta": 0.1,
    "prior_probability_shared_seasonality_alpha": 1,
    "prior_probability_shared_seasonality_beta": 1,
    "individual_fourier_terms": [
        {'seasonality_period_baseline': gmp.number_of_weeks_in_a_year,'number_of_fourier_components': 20}
    ],
    "shared_fourier_terms": [
        {'seasonality_period_baseline': gmp.number_of_weeks_in_a_year,'number_of_fourier_components': 10},
        {'seasonality_period_baseline': gmp.number_of_weeks_in_a_year/4,'number_of_fourier_components': 3},
        {'seasonality_period_baseline': gmp.number_of_weeks_in_a_year/12,'number_of_fourier_components': 1},
    ]
}

def generate_sorcerer_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        sampler_config: dict = sampler_config_default,
        model_config: dict = model_config_default
        ):
    sorcerer = SorcererModel(
        model_config = model_config,
        model_name = "SorcererModel",
        model_version = "v0.3.1"
        )
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_sorcerer_stacked_residuals'):
        training_data = df.iloc[:-fh]
        testing_data = df.iloc[-fh:].head(forecast_horizon)
        y_train_min = training_data[time_series_column_group].min()
        y_train_max = training_data[time_series_column_group].max()
        suppress_output(func = sorcerer.fit, training_data=training_data, sampler_config = sampler_config)
        (preds_out_of_sample, model_preds) = suppress_output(sorcerer.sample_posterior_predictive, test_data=testing_data)
        model_forecasts_unnormalized = pd.DataFrame(data = model_preds["target_distribution"].mean(("chain", "draw")), columns = time_series_column_group)
        model_forecasts = model_forecasts_unnormalized*(y_train_max-y_train_min)+y_train_min
        residuals.append((testing_data[time_series_column_group]-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle(f"./data/results/stacked_residuals_sorcerer_{sampler_config['sampler']}.pkl")

generate_sorcerer_stacked_residuals(sampler_config = sampler_config_MAP)
generate_sorcerer_stacked_residuals()