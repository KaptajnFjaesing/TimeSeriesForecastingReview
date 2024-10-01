"""
Created on Mon Sep 30 21:05:41 2024

@author: Jonas Petersen
"""

import numpy as np
import pymc as pm
import pandas as pd

from matplotlib import pyplot as plt

def simulate_ar(intercept, coef1, coef2, noise=0.3, *, warmup=10, steps=200):
    # We sample some extra warmup steps, to let the AR process stabilize
    draws = np.zeros(warmup + steps)
    # Initialize first draws at intercept
    draws[:2] = intercept
    for step in range(2, warmup + steps):
        draws[step] = (
            intercept
            + coef1 * draws[step - 1]
            + coef2 * draws[step - 2]
            + np.random.normal(0, noise)
        )
    # Discard the warmup draws
    return draws[warmup:]

steps = 200

# True parameters of the AR process
ar1_data1 = simulate_ar(
    intercept = 10,
    coef1 = -0.9,
    coef2 = -0.1,
    noise = 0.5,
    warmup = 20,
    steps = steps
    )

ar1_data2 = simulate_ar(
    intercept = 5,
    coef1 = 0.4,
    coef2 = 0.1,
    noise = 0.3,
    warmup = 20,
    steps = steps
    )

t = np.arange(steps)
df = pd.DataFrame({'time': t, 'value1': ar1_data1, 'value2': ar1_data2})
scale = 20

split = int(len(df)*0.77)
x_train = df['time'].values[:split]
y_train = df[['value1','value2']].values[:split]/scale

x_test = df['time'].values[split:]
y_test = df[['value1','value2']].values[split:]/scale

plt.figure(figsize=(12, 6))
plt.plot(x_train,y_train, label = "training data")
plt.plot(x_test,y_test, label = "test data")


#%%

coefs = 5

priors = {
    "coefs": {"mu": 1, "sigma": 1, "size": coefs},
    "init": {"mu": 1, "sigma": 1, "size": coefs-1},
}

number_of_time_series = y_train.shape[1]

coords = {
    "number_of_time_series": range(y_train.shape[1]),
    "coef_size": range(priors['coefs']['size']+1),
    "number_of_observations": range(y_train.shape[0])
    }

alpha = 2
beta = 0.1
with pm.Model(coords = coords) as AR_model1:
    y1 = pm.Data("y", y_train)

    rho = pm.Normal("coefs", mu = priors['coefs']['mu'], sigma= priors['coefs']['sigma'], shape = (number_of_time_series, priors['coefs']['size']))
    
    ar_precision_prior = pm.Gamma('ar_precision_prior', alpha = alpha, beta = beta)  
    init = pm.Normal.dist(mu = priors["init"]["mu"], sigma = priors["init"]["sigma"], shape = (number_of_time_series, priors['coefs']['size']-1))
    
    ar1 = pm.AR(
        "ar",
        rho = rho,
        sigma = 1/ar_precision_prior,
        init_dist = init,
        constant=True,
        steps=y1.shape[0] - (priors['coefs']['size'] - 1),
        dims = ['number_of_time_series', 'number_of_observations']
    ).T
    
    target_precision_prior = pm.Gamma('target_precision_prior', alpha = alpha, beta = beta)
    pm.Normal("target_distribution", mu = ar1, sigma = 1/target_precision_prior, observed = y1, dims=["number_of_observations",'number_of_time_series'])
    idata_ar = pm.sample(tune=100, draws=500, chains=1, return_inferencedata=True)
    posterior1 = pm.sample_posterior_predictive(idata_ar, predictions=True)


fig, ax = plt.subplots(figsize = (10, 3))
ax.set_title("Generated Autoregressive Timeseries", fontsize = 15)
ax.plot(x_train,y_train)
ax.plot(x_test,y_test)
ax.plot(x_train, posterior1.predictions["target_distribution"].mean(["chain", "draw"]).values)


#%%
forecast_horizon = len(x_test)
last_ar_draw = idata_ar.posterior['ar'].mean(('chain','draw')).values[:,-(priors['coefs']['size']-1):]

with pm.Model(coords = coords) as ar_test_model:
    ar_test_model.add_coords({"obs_id_fut_1": range(y_train.shape[0] - 1, y_train.shape[0] - 1+ forecast_horizon+1, 1)})
    ar_test_model.add_coords({"obs_id_fut": range(y_train.shape[0], y_train.shape[0]+forecast_horizon, 1)})

    rho = pm.Normal("coefs", mu = priors['coefs']['mu'], sigma = priors['coefs']['sigma'], shape = (number_of_time_series, priors['coefs']['size']))
    ar_precision_prior = pm.Gamma('ar_sigma_prior', alpha = alpha, beta = beta)
    initial_distribution = pm.DiracDelta.dist( last_ar_draw, shape = (number_of_time_series, priors['coefs']['size']-1))
    
    ar1_fut = pm.AR(
        "ar1_fut",
        init_dist = initial_distribution,
        rho = rho,
        sigma = 1/ar_precision_prior,
        constant = True,
        dims = ['number_of_time_series', 'obs_id_fut_1']
    ).T
    
    target_precision_prior = pm.Gamma('target_precision_prior', alpha = alpha, beta = beta )
    yhat_fut = pm.Normal("yhat_fut", mu = ar1_fut[:-1,:], sigma = 1/target_precision_prior, dims = ["obs_id_fut",'number_of_time_series'])
    posterior_predictive = pm.sample_posterior_predictive(idata_ar, var_names = ["yhat_fut"], predictions = True )

fig, ax = plt.subplots(figsize=(10, 3))
ax.set_title("Generated Autoregressive Timeseries", fontsize=15)
ax.plot(range(len(y_train)),y_train)
ax.plot(x_test,y_test)

ax.plot(x_train, posterior1.predictions["target_distribution"].mean(["chain", "draw"]).values)
ax.plot(x_test[:forecast_horizon], posterior_predictive.predictions['yhat_fut'].mean(["chain", "draw"]).values)
