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



training_split = int(len(df_passengers)*0.7)

normalization = 1000

x_train = df_passengers.index[:training_split].values/normalization
y_train = df_passengers['Passengers'].iloc[:training_split]/normalization


x_test = df_passengers.index[training_split:].values/normalization
y_test = df_passengers['Passengers'].iloc[training_split:]/normalization



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

with pm.Model() as prophet_model:
    t_pt = pm.Data('t', x_train, dims = ['n_obs'])

    # We have monthly data, so p=12 leads to annual seasonality
    A, s, X = generate_features(t_pt, max(x_train), n_changepoints=10, n_fourier=6, p=12)
    
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
    y_hat = pm.Normal('y_hat', mu=mu, sigma=sigma, observed=y_train, dims = ['n_obs'])
    
    idata = pm.sample(tune=100, draws=5000, chains=1)
    
# %%

with prophet_model:
    pm.set_data({'t':x_test})
    posterior_predictive = pm.sample_posterior_predictive(idata, predictions=True)
    
predictions = posterior_predictive.predictions['y_hat'].mean((('chain'))).values

#%%
plt.figure()
plt.plot(x_test,predictions.mean(axis = 0), color = 'black')
for i in predictions:
    plt.plot(x_test,i, alpha = 0.01, color = 'red')
plt.plot(x_train,y_train)
plt.plot(x_test,y_test, color = "blue")
