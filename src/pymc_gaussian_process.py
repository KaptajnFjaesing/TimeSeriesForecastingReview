# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:23:30 2024

@author: AWR
"""

#%% Import section
from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az


df_passengers = load_passengers_data()  # Load the data

#%% Model setup
t=np.arange(len(df_passengers))[:,None]
y = df_passengers['Passengers'].values#[:,None]
with pm.Model() as gp_model:
    
    lp_smooth = pm.Gamma('lp_smooth',alpha=2,beta=0.5)
    seasonal_eta_prior = pm.HalfNormal("seasonal_eta", sigma=5)
    
    cov_func_year = seasonal_eta_prior**2 * pm.gp.cov.Periodic(1, period=12, ls=lp_smooth)
    gp_seasonal = pm.gp.Latent(cov_func=cov_func_year)
    seasonal = gp_seasonal.prior('seasonal',X=t)
    
    lt_smooth = pm.Gamma('lt_smooth',alpha=2,beta=1)
    trend_eta_prior = pm.HalfNormal("trend_eta", sigma=10)  # Prior for amplitude
    
    cov_trend = trend_eta_prior**2 * pm.gp.cov.ExpQuad(1,ls=lt_smooth)
    gp_trend = pm.gp.Latent(cov_func=cov_trend)
    trend = gp_trend.prior('trend',X=t)
    
    f = trend + seasonal
    
    sigma_ = pm.HalfCauchy('sigma_',5)
    
    obs = pm.Normal('obs', mu=f, sigma=sigma_, observed=y)
    
#%% sample posterior

with gp_model:
    #posterior = pm.sample(tune=500,draws=500,chains=4)
    approx = pm.fit(n=20000)
    
#%% Convergence plot
plt.plot(approx.hist)
#%% sample posterior predictive
with gp_model:
    samples = approx.sample(1000)
with gp_model:
    pp = pm.sample_posterior_predictive(samples)
#%% Trace plot
az.plot_trace(samples)
#%%
plt.plot(y)
plt.plot(pp.posterior_predictive['obs'].mean(('chain','draw')))