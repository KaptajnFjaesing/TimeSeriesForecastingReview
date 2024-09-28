
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

_,df,_ = load_m5_weekly_store_category_sales_data()

nan_count = [0, 102, 73, 17, 37, 42, 9, 4, 0, 8]
time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]

# Assign NaNs to the start of each column dynamically
for col, n in zip(time_series_column_group, nan_count):
    df.loc[:n-1, col] = np.nan

#%%

forecast_horizon = 30

training_data = df.iloc[:-forecast_horizon]
test_data = df.iloc[-forecast_horizon:]

def add_linear_term(
    model: pm.Model,
    x: np.array,
    baseline_slope: float,
    baseline_bias: float,
    trend_name: str,
    number_of_time_series: int,
    number_of_trend_changepoints: int,
    delta_mu_prior: float,
    delta_b_prior: float,
    m_sigma_prior: float,
    k_sigma_prior: float
):
    with model:
        s = pt.tensor.linspace(0, 1, number_of_trend_changepoints + 2)[1:-1] # max(x) for input is by definition 1
        A = (x[:, None] > s) * 1.
        k = pm.Normal(f'{trend_name}_k', mu=baseline_slope, sigma=k_sigma_prior, shape=number_of_time_series)
        delta = pm.Laplace(f'{trend_name}_delta', mu=delta_mu_prior, b=delta_b_prior, shape=(number_of_time_series, number_of_trend_changepoints))
        m = pm.Normal(f'{trend_name}_m', mu=baseline_bias, sigma=m_sigma_prior, shape=number_of_time_series)
    return (k + pm.math.dot(A, delta.T)) * x[:, None] + m + pm.math.dot(A, (-s * delta).T)

def add_fourier_term(
    model: pm.Model,
    x: np.array,
    number_of_fourier_components: int,
    name: str,
    dimension: int,
    seasonality_period_baseline: float,
    fourier_sigma_prior: float,
    fourier_mu_prior: float
):
    with model:
        fourier_coefficients = pm.Normal(f'fourier_coefficients_{name}', mu=fourier_mu_prior, sigma = fourier_sigma_prior, shape=(2 * number_of_fourier_components, dimension))
        fourier_features = create_fourier_features(x=x, number_of_fourier_components=number_of_fourier_components, seasonality_period=seasonality_period_baseline)
    return pm.math.sum(fourier_features * fourier_coefficients[None, :, :], axis=1)

def create_fourier_features(
    x: np.array,
    number_of_fourier_components: int,
    seasonality_period: float
) -> np.array:
    frequency_component = pt.tensor.as_tensor_variable(2 * np.pi * (np.arange(number_of_fourier_components) + 1) * x[:, None])
    t = frequency_component[:, :, None] / seasonality_period
    return pm.math.concatenate((pt.tensor.cos(t), pt.tensor.sin(t)), axis=1)

# %%

# Sorcerer
sampler_config = {
    "draws": 1000,
    "tune": 200,
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

extended_training_data = extend_training_data(
        df_training = training_data,
        forecast_horizon = forecast_horizon
        )

X_training_min = extended_training_data['date'].astype('int64').min()//10**9
X_training_max = extended_training_data['date'].astype('int64').max()//10**9
X = ((extended_training_data['date'].astype('int64')//10**9-X_training_min)/(X_training_max-X_training_min))


Y_training_min = extended_training_data[time_series_column_group].min()
Y_training_max = extended_training_data[time_series_column_group].max()
Y = ((extended_training_data[time_series_column_group]-Y_training_min)/(Y_training_max-Y_training_min))

X_test = X.iloc[-forecast_horizon:]
Y_test = ((test_data[time_series_column_group]-Y_training_min)/(Y_training_max-Y_training_min))

mask = extended_training_data[time_series_column_group]
mask = mask.where(mask.isna(), 1)
X_matrix = mask.multiply(X, axis=0)

baseline_slope = (Y.apply(lambda col: col[col.last_valid_index()])-Y.apply(lambda col: col[col.first_valid_index()])) / (X_matrix.apply(lambda col: col[col.last_valid_index()])-X_matrix.apply(lambda col: col[col.first_valid_index()]))
baseline_bias = Y.apply(lambda col: col[col.first_valid_index()])

number_of_time_series = len(time_series_column_group)

coords = {
    'number_of_time_series': range(number_of_time_series)
    }

dX = X[1]-X[0]

#%%
with pm.Model(coords = coords) as model:
    
    x = pm.Data('input', X, dims='number_of_input_observations')

    linear_term = add_linear_term(
        x=x,
        baseline_slope=baseline_slope,
        baseline_bias=baseline_bias,
        trend_name='linear',
        number_of_time_series=number_of_time_series,
        number_of_trend_changepoints=model_config["number_of_individual_trend_changepoints"],
        delta_mu_prior=model_config["delta_mu_prior"],
        delta_b_prior=model_config["delta_b_prior"],
        m_sigma_prior=model_config["m_sigma_prior"],
        k_sigma_prior=model_config["k_sigma_prior"],
        model=model
    )
    
    if len(model_config["individual_fourier_terms"]) > 0:
        seasonality_individual = pm.math.sum([
            add_fourier_term(
                x=x,
                number_of_fourier_components= term['number_of_fourier_components'],
                name=f"seasonality_individual_{round(term['seasonality_period_baseline'],2)}",
                dimension=number_of_time_series,
                seasonality_period_baseline=term['seasonality_period_baseline']*dX,
                model=model,
                fourier_sigma_prior = model_config["fourier_sigma_prior"],
                fourier_mu_prior = model_config["fourier_mu_prior"]
            ) for term in model_config["individual_fourier_terms"]
            ], axis = 0)
    else:
        print("asd")
    if len(model_config["shared_fourier_terms"]) > 0:
        shared_seasonalities = pm.math.concatenate([
            add_fourier_term(
                x=x,
                number_of_fourier_components=term['number_of_fourier_components'],
                name=f"seasonality_shared_{round(term['seasonality_period_baseline'], 2)}",
                dimension=1,
                seasonality_period_baseline=term['seasonality_period_baseline'] * dX,
                model=model,
                fourier_sigma_prior=model_config["fourier_sigma_prior"],
                fourier_mu_prior=model_config["fourier_mu_prior"]
            ) for term in model_config["shared_fourier_terms"]
        ], axis=1)
        
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
    
    target_mean = (
        linear_term +
        seasonality_individual +
        shared_seasonality
    )
    
    precision_target = pm.Gamma(
    'precision_target_distribution',
    alpha = model_config["precision_target_distribution_prior_alpha"],
    beta = model_config["precision_target_distribution_prior_beta"],
    dims = 'number_of_time_series'
    )
    pm.Normal('target_distribution', mu=target_mean, sigma=1/precision_target, observed=Y, dims=['number_of_input_observations', 'number_of_time_series'])

# %%

sampler_config = {
    "draws": 600,
    "tune": 200,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "verbose": True
}

with model:
    idata = pm.sample(step = pm.NUTS(), **{k: v for k, v in sampler_config.items() if (k != 'sampler' and k != 'verbose')})


# %%
with model:
    pm.set_data({'input': X})
    posterior_predictive = pm.sample_posterior_predictive(idata, predictions=True, progressbar = True)

preds_out_of_sample = posterior_predictive.predictions_constant_data.sortby('input')['input']
model_preds = posterior_predictive.predictions.sortby(preds_out_of_sample)

# %%




column_names = [x for x in df.columns if 'HOUSEHOLD' in x]

hdi_values = az.hdi(model_preds)["target_distribution"].transpose("hdi", ...)

n_cols = 2
n_rows = int(np.ceil(len(column_names) / n_cols))  # Number of rows needed
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
axs = axs.flatten()
for i in range(len(column_names)):
    ax = axs[i]
    ax.plot(X, Y[Y.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(X_test, Y_test[Y_test.columns[i]], color = 'black',  label='Test Data')
    ax.plot(preds_out_of_sample, (model_preds["target_distribution"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    ax.fill_between(
        preds_out_of_sample.values,
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