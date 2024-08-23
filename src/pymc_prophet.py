from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytensor as pt


"""
TLP_functions.py are in the VICKP project.

 - Input also the week of the year

"""

df_passengers = load_passengers_data()  # Load the data

def create_piecewise_trend(t, t_max, n_changepoints):    
    s = np.linspace(0, t_max, n_changepoints+2)[1:-1]
    A = (t[:, None] > s)*1
    
    return A, s

def create_fourier_features(t, n, p = 365.25):
    x = 2 * np.pi * (np.arange(n)+1) * t[:, None] / p
    return pm.math.concatenate((np.cos(x), np.sin(x)), axis = 1)

def generate_features(t, t_max, n_changepoints=10, n_fourier=6, p=365.25):
    A, s = create_piecewise_trend(t, t_max, n_changepoints)
    X = create_fourier_features(t, n_fourier, p)
    
    return A, s, X

# %%
t = np.arange(df_passengers.shape[0])

# This value isn't on the computation graph (it's not computed dynamically from `t_pt`)
t_max = max(t)

with pm.Model() as prophet_model:
    t_pt = pm.Data('t', t)

    # We have monthly data, so p=12 leads to annual seasonality
    A, s, X = generate_features(t_pt, t_max, n_changepoints=10, n_fourier=6, p=12)
    
    initial_slope = pm.Normal('initial_slope')
    initial_intercept = pm.Normal('initial_intercept')
    
    # n_changepoint offsets terms to build the peicewise trend
    deltas = pm.Normal('offset_delta', shape=(10,))
        
    intercept = initial_intercept + ((-s * A) * deltas).sum(axis=1)
    slope = initial_slope + (A * deltas).sum(axis=1)
    
    # n_fourier * 2 seasonal coefficients
    beta = pm.Normal('beta', size=12)
    
    mu = pm.Deterministic('mu', intercept + slope * t_pt+ X @ beta)
    sigma = pm.Exponential('sigma', 1)
    y_hat = pm.Normal('y_hat', mu=mu, sigma=sigma, observed=df_passengers['Passengers'].values.ravel(), shape=t_pt.shape[0])
    
    idata = pm.sample(tune=500, draws=1000, chains=1)
    
# %%

with prophet_model:
    last_t = t[-1]
    forcast_t = np.arange(last_t, last_t + 36)
    pm.set_data({'t':forcast_t})
    posterior_predictive = pm.sample_posterior_predictive(idata, extend_inferencedata=True, predictions=True)
    

# %%
predictions = posterior_predictive.predictions['y_hat'].mean((('chain'))).values

plt.figure()
plt.plot(forcast_t,predictions.mean(axis = 0), color = 'black')
for i in predictions:
    plt.plot(forcast_t,i, alpha = 0.01, color = 'red')
plt.plot(idata.observed_data['y_hat'])
