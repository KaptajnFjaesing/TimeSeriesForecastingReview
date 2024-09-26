# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:21:35 2024

@author: petersen.jonas
"""

import numpy as np
import pandas as pd
from src.data_loader import load_m5_weekly_store_category_sales_data
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Load data
_, df, _ = load_m5_weekly_store_category_sales_data()

# %%

# Test/Training split
split = 50
training_data = df.iloc[:-split]
test_data = df.iloc[-split:]

# Training data
target_columns = training_data.columns[2]
sales = training_data[target_columns]
x_train = np.arange(len(sales))[:,None]

# Standardize training data (optional)
sales_mean = np.mean(sales)
sales_std = np.std(sales)
y_train = ((sales - sales_mean) / sales_std).values[:,None]

n_time_series = len(training_data.columns[1:])
#%%

# Gaussian Process model
with pm.Model() as gp_model:
    x = pm.Data('x', x_train, dims= ['n_obs', 'dummy'])
    y = pm.Data('y', y_train, dims= ['n_obs_target', 'n_time_series'])
    # Covariance function (shared across all time series)
    cov_func = (
        pm.gp.cov.Periodic(input_dim = 1, period=52, ls=52) +
        pm.gp.cov.Periodic(input_dim = 1, period=13, ls=13) +
        pm.gp.cov.Periodic(input_dim = 1, period=4, ls=4) +
        pm.gp.cov.ExpQuad(input_dim = 1, ls=52)
    )
    gp = pm.gp.Latent(cov_func=cov_func)
 
    f = gp.prior("f", X=x, dims = 'n_obs')
    y_obs = pm.Normal("y_obs", mu=f[:,None], sigma=0.01, observed=y, dims = ['n_obs', 'n_time_series'] )
    trace = pm.sample(tune=100, draws=200, chains=1, cores=1, step=pm.NUTS(), nuts_sampler="numpyro")
    

test_week = np.arange(len(sales), len(sales) + len(test_data))

# Predictive sampling for test data
with gp_model:
    f_pred = gp.conditional("f_pred", Xnew=test_week[:, None])
    pred_samples = pm.sample_posterior_predictive(trace, var_names=["f_pred"])


#%%

pred_mean = pred_samples.posterior_predictive['f_pred'].mean(('chain', 'draw')).values * sales_std + sales_mean

test_sales = test_data[target_columns].values

plt.figure()
plt.plot(x_train,sales, label = "training data")
plt.plot(test_week, test_sales, label = "test data")
plt.plot(test_week, pred_mean, label = "model")
plt.legend()

# %%
# Plot the results
az.plot_trace(trace)


