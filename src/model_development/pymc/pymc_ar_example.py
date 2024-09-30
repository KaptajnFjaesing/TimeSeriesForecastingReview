"""
Created on Mon Sep 30 20:48:42 2024

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

training_data = ar1_data[:split]
test_data = ar1_data[split:]

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

t_data = list(range(len(training_data)))

with pm.Model() as AR:
    t = pm.Data("t", t_data, dims="obs_id")
    y = pm.Data("y", training_data, dims="obs_id2")

    rho = pm.Normal("coefs", priors["coefs"]["mu"], priors["coefs"]["sigma"])
    sigma = pm.HalfNormal("sigma", priors["sigma"])
    init = pm.Normal.dist(priors["init"]["mu"], priors["init"]["sigma"], size=priors["init"]["size"])
    ar1 = pm.AR(
        "ar",
        rho = rho,
        sigma=sigma,
        init_dist=init,
        constant=True,
        steps=t.shape[0] - (priors["coefs"]["size"] - 1),
        dims="obs_id",
    )

    outcome = pm.Normal("likelihood", mu=ar1, sigma=sigma, observed=y, dims="obs_id")
    idata_ar = pm.sample(tune=100, draws=200, chains=1)

with AR:
    AR.add_coords({"obs_id_fut_1": range(training_data.shape[0] - 1, len(ar1_data), 1)})
    AR.add_coords({"obs_id_fut": range(training_data.shape[0], len(ar1_data), 1)})
    ar1_fut = pm.AR(
        "ar1_fut",
        init_dist=pm.DiracDelta.dist(ar1[..., -1]),
        rho=rho,
        sigma=sigma,
        constant=True,
        dims="obs_id_fut_1",
    )
    yhat_fut = pm.Normal("yhat_fut", mu=ar1_fut[1:], sigma=sigma, dims="obs_id_fut")
    idata_preds = pm.sample_posterior_predictive(idata_ar, var_names=["likelihood", "yhat_fut"], predictions=True)

fig, ax = plt.subplots(figsize=(10, 3))
ax.set_title("Generated Autoregressive Timeseries", fontsize=15)
ax.plot(range(len(training_data)),training_data)
ax.plot(range(len(training_data),len(ar1_data)),test_data)

idata_preds.predictions.yhat_fut.mean(["chain", "draw"]).plot(color="cyan", label="Predicted Mean Realisation"
)