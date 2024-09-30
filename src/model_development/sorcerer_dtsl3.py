
"""
Created on Mon Sep  9 20:22:19 2024

@author: Jonas Petersen
"""
import numpy as np
import arviz as az
import pymc as pm
import pytensor as pt
import matplotlib.pyplot as plt
from src.data_loader import load_m5_weekly_store_category_sales_data


def create_fourier_features(
    x: np.array,
    number_of_fourier_components: int,
    seasonality_period: float
) -> np.array:
    frequency_component = pt.tensor.as_tensor_variable(2 * np.pi * (np.arange(number_of_fourier_components) + 1) * x[:, None])
    t = frequency_component[:, :, None] / seasonality_period
    return pm.math.concatenate((pt.tensor.cos(t), pt.tensor.sin(t)), axis=1)

_,df,_ = load_m5_weekly_store_category_sales_data()

nan_count = [0, 102, 73, 17, 180, 42, 9, 4, 0, 8]
time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]

# Assign NaNs to the start of each column dynamically
for col, n in zip(time_series_column_group, nan_count):
    df.loc[:n-1, col] = np.nan


sampler_config = {
    "draws": 100,
    "tune": 50,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "verbose": True
}

number_of_weeks_in_a_year = 52.1429

model_config = {
    "number_of_individual_trend_changepoints": 40,
    "delta_mu_prior": 0,
    "delta_b_prior": 0.1,
    "m_sigma_prior": 0.2,
    "k_sigma_prior": 0.2,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior" : 1,
    "precision_target_distribution_prior_alpha": 10,
    "precision_target_distribution_prior_beta": 0.1,
    "prior_probability_shared_seasonality_alpha": 1,
    "prior_probability_shared_seasonality_beta": 1,
    "rho_mu_prior": 0,
    "rho_sigma_prior": 1,
    "tau_lam_prior": 0.1,
    "rho_init_mu_prior": 0,
    "rho_init_sigma_prior": 10,
    "rho_init_size": 1,
    "number_of_autoregressive_terms": 2,
    "individual_fourier_terms": [
        {'seasonality_period_baseline': number_of_weeks_in_a_year,'number_of_fourier_components': 20}
    ],
    "shared_fourier_terms": [
        {'seasonality_period_baseline': number_of_weeks_in_a_year,'number_of_fourier_components': 10},
        {'seasonality_period_baseline': number_of_weeks_in_a_year/4,'number_of_fourier_components': 1},
        {'seasonality_period_baseline': number_of_weeks_in_a_year/12,'number_of_fourier_components': 1},
    ]
}

#%%

import pandas as pd

def extend_training_data(
        df_training: pd.DataFrame,
        forecast_horizon: int
        ):
    time_delta = training_data['date'].diff().dropna().unique()[0]
    new_dates = pd.date_range(start=training_data['date'].iloc[-1] + time_delta, periods=forecast_horizon, freq=time_delta)
    new_rows = pd.DataFrame({
        'date': new_dates
    })
    return pd.concat([training_data, new_rows], ignore_index=True)

def generate_XY(
        df: pd.DataFrame,
        time_series_column_group: list[str]
        ):
    X_training_min = df['date'].astype('int64').min()//10**9
    X_training_max = df['date'].astype('int64').max()//10**9
    
    X = ((df['date'].astype('int64')//10**9-X_training_min)/(X_training_max-X_training_min))
    
    Y_training_min = df[time_series_column_group].min()
    Y_training_max = df[time_series_column_group].max()
    Y = ((df[time_series_column_group]-Y_training_min)/(Y_training_max-Y_training_min))
    
    return X,Y

forecast_horizon = 30

training_data = df.iloc[:-forecast_horizon]
test_data = df.iloc[-forecast_horizon:]

extended_training_data = extend_training_data(
        df_training = training_data,
        forecast_horizon = forecast_horizon
        )


X, Y = generate_XY(df = training_data, time_series_column_group = time_series_column_group)
X_extended, Y_extended = generate_XY(df = extended_training_data, time_series_column_group = time_series_column_group)

X_training_min = training_data['date'].astype('int64').min()//10**9
X_training_max = training_data['date'].astype('int64').max()//10**9

X_test = ((test_data['date'].astype('int64')//10**9-X_training_min)/(X_training_max-X_training_min))

Y_training_min = training_data[time_series_column_group].min()
Y_training_max = training_data[time_series_column_group].max()
#X_test = X_extended.iloc[-forecast_horizon:]
Y_test = ((test_data[time_series_column_group]-Y_training_min)/(Y_training_max-Y_training_min))

mask = extended_training_data[time_series_column_group]
mask = mask.where(mask.isna(), 1)
X_matrix = mask.multiply(X, axis=0)

baseline_slope = (Y.apply(lambda col: col[col.last_valid_index()])-Y.apply(lambda col: col[col.first_valid_index()])) / (X_matrix.apply(lambda col: col[col.last_valid_index()])-X_matrix.apply(lambda col: col[col.first_valid_index()]))
baseline_bias = Y.apply(lambda col: col[col.first_valid_index()])





dX = X[1]-X[0]

def ar_term(
        model_config,
        number_of_time_series,
        number_of_observations,
        model
        ):
    with model:
        rho = pm.Normal(
            "rho",
            mu=model_config["rho_mu_prior"],
            sigma=model_config["rho_sigma_prior"],
            shape = (number_of_time_series, model_config["number_of_autoregressive_terms"] + 1)  # +1 for the constant term
        )
        
        tau = pm.Exponential(
            "tau", 
            lam=model_config["tau_lam_prior"], 
            shape = number_of_time_series  # A separate noise parameter for each time series
        )
        
        # Initial values for the AR process, one set for each time series
        init = pm.Normal.dist(
            mu=model_config["rho_init_mu_prior"],
            sigma=model_config["rho_init_sigma_prior"],
            shape=(number_of_time_series, model_config["number_of_autoregressive_terms"])  # One set of initial values per time series
        )
        
        # Apply the AR process independently for each time series
        return pm.AR(
            "ar",
            rho=rho,
            tau=tau,
            init_dist=init,
            constant=True,
            shape = (number_of_time_series, number_of_observations)  # Output: (time steps, number of time series)
        ).T

def generate_mean(
        X,
        baseline_slope,
        baseline_bias,
        model_config,
        model,
        number_of_time_series
        ):
    with model:
        
        number_of_observations = X.shape[0]
        
        x = pm.Data('input_training', X, dims = 'number_of_observations')
        
        s = pt.tensor.linspace(0, 1, model_config["number_of_individual_trend_changepoints"] + 2)[1:-1] # max(x) for input is by definition 1
        A = (x[:, None] > s) * 1.
        k = pm.Normal('trend_k', mu=baseline_slope, sigma=model_config["k_sigma_prior"], shape=number_of_time_series)
        delta = pm.Laplace('trend_delta', mu=model_config["delta_mu_prior"], b=model_config["delta_b_prior"], shape=(number_of_time_series, model_config["number_of_individual_trend_changepoints"]))
        m = pm.Normal('trend_m', mu=baseline_bias, sigma=model_config["m_sigma_prior"], shape=number_of_time_series)
        trend_term = (k + pm.math.dot(A, delta.T)) * x[:, None] + m + pm.math.dot(A, (-s * delta).T)
        
        seasonality_individual = pm.math.zeros((number_of_observations, number_of_time_series))
        for term in model_config["individual_fourier_terms"]:
            number_of_fourier_components = term['number_of_fourier_components']
            seasonality_period_baseline = term['seasonality_period_baseline'] * dX
            
            # Define distinct variable names for coefficients
            fourier_coefficients = pm.Normal(
                f'fourier_coefficients_{round(seasonality_period_baseline, 2)}',
                mu=model_config["fourier_mu_prior"],
                sigma=model_config["fourier_sigma_prior"],
                shape=(2 * number_of_fourier_components, number_of_time_series)
            )
            
            # Create Fourier features
            fourier_features = create_fourier_features(
                x=x,
                number_of_fourier_components=number_of_fourier_components,
                seasonality_period=seasonality_period_baseline
            )
            seasonality_individual += pm.math.sum(fourier_features * fourier_coefficients[None, :, :], axis=1)
        
        if len(model_config["shared_fourier_terms"]) > 0:
            shared_seasonalities = []
            for term in model_config["shared_fourier_terms"]:
                number_of_fourier_components = term['number_of_fourier_components']
                seasonality_period_baseline = term['seasonality_period_baseline'] * dX
                
                # Define distinct variable names for coefficients
                fourier_coefficients = pm.Normal(
                    f'fourier_coefficients_shared_{round(seasonality_period_baseline, 2)}',
                    mu=model_config["fourier_mu_prior"],
                    sigma=model_config["fourier_sigma_prior"],
                    shape=(2 * number_of_fourier_components, 1)
                )
                
                # Create Fourier features
                fourier_features = create_fourier_features(
                    x=x,
                    number_of_fourier_components=number_of_fourier_components,
                    seasonality_period=seasonality_period_baseline
                )
                
                # Calculate the seasonal term and store it
                shared_seasonality_term = pm.math.sum(fourier_features * fourier_coefficients[None, :, :], axis=1)
                shared_seasonalities.append(shared_seasonality_term)
        
            # Combine all shared seasonal terms into one array
            shared_seasonalities = pm.math.concatenate(shared_seasonalities, axis=1)
            
            prior_probability_shared_seasonality = pm.Beta(
                'prior_probability_shared_seasonality',
                alpha=model_config["prior_probability_shared_seasonality_alpha"],
                beta=model_config["prior_probability_shared_seasonality_beta"],
                shape = number_of_time_series
                )
            include_seasonality = pm.Bernoulli(
                'include_seasonality',
                p=prior_probability_shared_seasonality,
                shape=(len(model_config["shared_fourier_terms"]), number_of_time_series)
            )
            shared_seasonality = pm.math.dot(shared_seasonalities, include_seasonality)
        else:
            shared_seasonality = 0

        return trend_term + seasonality_individual + shared_seasonality



# %%
number_of_time_series = len(time_series_column_group)

coords = {
    'number_of_time_series': range(number_of_time_series),
    'number_of_observations': range(X.shape[0])
    }

with pm.Model(coords = coords) as model_training:


    number_of_observations = X.shape[0]
    
    mu_trend_seasonality_training = generate_mean(
            X = X,
            baseline_slope = baseline_slope,
            baseline_bias = baseline_bias,
            model_config = model_config,
            model = model_training,
            number_of_time_series = number_of_time_series
            )
    
    rho = pm.Normal(
        "rho",
        mu=model_config["rho_mu_prior"],
        sigma=model_config["rho_sigma_prior"],
        shape = (number_of_time_series, model_config["number_of_autoregressive_terms"] + 1)  # +1 for the constant term
    )
    
    tau = pm.Exponential(
        "tau", 
        lam=model_config["tau_lam_prior"], 
        shape = number_of_time_series  # A separate noise parameter for each time series
    )
    
    # Initial values for the AR process, one set for each time series
    init = pm.Normal.dist(
        mu=model_config["rho_init_mu_prior"],
        sigma=model_config["rho_init_sigma_prior"],
        shape=(number_of_time_series, model_config["number_of_autoregressive_terms"])  # One set of initial values per time series
    )

    mu_autoregressive_term_training = pm.AR(
        "ar",
        rho=rho,
        tau=tau,
        init_dist=init,
        constant=True,
        dims = ['number_of_time_series', 'number_of_observations']
    ).T

    mu_training = mu_trend_seasonality_training + mu_autoregressive_term_training
    
    precision_target = pm.Gamma(
    'precision_target_distribution',
    alpha = model_config["precision_target_distribution_prior_alpha"],
    beta = model_config["precision_target_distribution_prior_beta"],
    shape = number_of_time_series
    )
    pm.Normal('target_distribution', mu = mu_training, sigma=1/precision_target, observed=Y, dims = ['number_of_observations', 'number_of_time_series'])
    idata = pm.sample(step = pm.NUTS(), **{k: v for k, v in sampler_config.items() if (k != 'sampler' and k != 'verbose')})

# %%

test_coords = {
    'number_of_time_series': range(number_of_time_series),
    'number_of_observations': range(X_test.shape[0])
    }

with pm.Model(coords = test_coords) as model_test:
    number_of_time_series = baseline_bias.shape[0]
    number_of_observations = X_test.shape[0]
    mu_trend_seasonality_test = generate_mean(
            X = X_test,
            baseline_slope = baseline_slope,
            baseline_bias = baseline_bias,
            model_config = model_config,
            model = model_test,
            number_of_time_series = number_of_time_series
            )

    mu_autoregressive_term_test = pm.AR(
        "ar_fut",
        init_dist=pm.DiracDelta.dist(mu_autoregressive_term_training[..., -1]),
        rho=rho,
        tau=tau,
        constant=True,
        dims = ['number_of_time_series', 'number_of_observations']
    ).T
    
    mu_test = mu_trend_seasonality_test + mu_autoregressive_term_test

    precision_target_test = pm.Gamma(
    'precision_target_distribution',
    alpha = model_config["precision_target_distribution_prior_alpha"],
    beta = model_config["precision_target_distribution_prior_beta"],
    shape = number_of_time_series
    )
    
    pm.Normal('predictions', mu = mu_test, sigma=1/precision_target_test, dims = ['number_of_observations', 'number_of_time_series'])

    posterior_predictive = pm.sample_posterior_predictive(
        idata,
        predictions = True,
        var_names=['predictions']
        )


#%%

column_names = [x for x in df.columns if 'HOUSEHOLD' in x]

hdi_values = az.hdi(posterior_predictive.predictions)["predictions"].transpose("hdi", ...)

n_cols = 2
n_rows = int(np.ceil(len(column_names) / n_cols))  # Number of rows needed
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
axs = axs.flatten()
for i in range(len(column_names)):
    ax = axs[i]
    ax.plot(X, Y[Y.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(X_test, Y_test[Y_test.columns[i]], color = 'black',  label='Test Data')
    ax.plot(X_test, (posterior_predictive.predictions["predictions"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    ax.fill_between(
        X_test,
        hdi_values[0].values[:,i],
        hdi_values[1].values[:,i],
        color= 'blue',
        alpha=0.4
    )
    ax.set_title(column_names[i])
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()



# %% DEVELOPMENT


with pm.Model() as model:
    number_of_observations = X.shape[0]
    number_of_time_series = 10
    
    
    rho = pm.Normal("rho",
                      mu = model_config["rho_mu_prior"],
                      sigma= model_config["rho_sigma_prior"],
                      shape = model_config["number_of_autoregressive_terms"]+1
                      )
    tau = pm.Exponential("tau", lam = model_config["tau_lam_prior"])
    init = pm.Normal.dist(
        mu = model_config["rho_init_mu_prior"],
        sigma = model_config["rho_init_sigma_prior"],
        shape = model_config["rho_init_size"]
    )
    autoregressive_term = pm.AR(
        "ar",
        rho = rho,
        tau=tau,
        init_dist = init,
        constant=True,
        shape = (number_of_observations, number_of_time_series)
    )

#%%



with pm.Model() as model:
    number_of_observations = X.shape[0]  # Total number of time steps/observations
    number_of_time_series = 10  # Total number of time series
    number_of_autoregressive_terms = model_config["number_of_autoregressive_terms"]

    # Define a separate set of rho (AR coefficients) for each time series
    hest = ar_term(
            model_config = model_config,
            number_of_time_series = number_of_time_series,
            number_of_observations = number_of_observations,
            model = model
            )

# This model now defines a separate autoregressive process for each time series.

