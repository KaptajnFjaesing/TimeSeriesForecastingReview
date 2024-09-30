"""
Created on Mon Sep 30 21:05:41 2024

@author: Jonas Petersen
"""

import numpy as np
import pymc as pm

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


# True parameters of the AR process
ar1_data = simulate_ar(10, -0.9, 0)
split = int(0.8*len(ar1_data))

training_data = ar1_data[:split].reshape(-1,1)
test_data = ar1_data[split:].reshape(-1,1)

fig, ax = plt.subplots(figsize=(10, 3))
ax.set_title("Generated Autoregressive Timeseries", fontsize=15)
ax.plot(range(len(training_data)),training_data)
ax.plot(range(len(training_data),len(ar1_data)),test_data)


#%%


priors = {
    "coefs": {"mu": [20, 1], "sigma": [1, 1], "size": 2},
    "sigma": 8,
    "init": {"mu": 9, "sigma": 0.1, "size": 1},
}

number_of_time_series = 1
coef_size = 2

coords = {
    "number_of_time_series": range(training_data.shape[1]),
    "coef_size": range(coef_size+1),
    "number_of_observations": range(training_data.shape[0])
    }

prediction_length = len(training_data)
t_data = list(range(len(training_data)))

with pm.Model(coords = coords) as AR_model1:
    y1 = pm.Data("y", training_data)

    rho = pm.Normal("coefs", mu = priors['coefs']['mu'], sigma= priors['coefs']['sigma'], shape = (number_of_time_series, priors['coefs']['size']))
    sigma = pm.HalfNormal("sigma", priors["sigma"])
    init = pm.Normal.dist(mu = priors["init"]["mu"], sigma = priors["init"]["sigma"], shape = (number_of_time_series, priors['coefs']['size']-1))
    
    ar1 = pm.AR(
        "ar",
        rho = rho,
        sigma = sigma,
        init_dist = init,
        constant=True,
        steps=y1.shape[0] - (priors['coefs']['size'] - 1),
        dims = ['number_of_time_series', 'number_of_observations']
    ).T
    pm.Normal("target_distribution",mu=ar1,sigma=sigma,observed=y1,dims=["number_of_observations",'number_of_time_series'])
    idata_ar = pm.sample(tune=100, draws=200, chains=1)


#%%
forecast_horizon = 20
with pm.Model(coords = coords) as ar_test_model:
    ar_test_model.add_coords({"obs_id_fut_1": range(training_data.shape[0] - 1, training_data.shape[0] - 1+ forecast_horizon+1, 1)})
    ar_test_model.add_coords({"obs_id_fut": range(training_data.shape[0], training_data.shape[0]+forecast_horizon, 1)})
    
    rho = pm.Normal("coefs",
                      mu = priors['coefs']['mu'],
                      sigma= priors['coefs']['sigma'],
                      shape = (number_of_time_series, priors['coefs']['size'])
                      )

    initial_distribution = pm.DiracDelta.dist(
        ar1[-(priors['coefs']['size']-1):, :].T,  
        shape = (number_of_time_series,priors['coefs']['size']-1)
    )
    sigma = pm.HalfNormal("sigma", priors["sigma"])
    ar1_fut = pm.AR(
        "ar1_fut",
        init_dist = initial_distribution,
        rho = rho,
        sigma = sigma,
        constant = True,
        dims = ['number_of_time_series', 'obs_id_fut_1']
    ).T

    yhat_fut = pm.Normal("yhat_fut",mu=ar1_fut[1:,:],sigma=sigma,dims=["obs_id_fut",'number_of_time_series'])
    posterior_predictive = pm.sample_posterior_predictive(idata_ar, var_names = ["yhat_fut"], predictions = True )


fig, ax = plt.subplots(figsize=(10, 3))
ax.set_title("Generated Autoregressive Timeseries", fontsize=15)
ax.plot(range(len(training_data)),training_data)
ax.plot(range(len(training_data),len(ar1_data)),test_data)

posterior_predictive.predictions.yhat_fut.mean(["chain", "draw"]).plot(color="cyan", label="Predicted Mean Realisation")
