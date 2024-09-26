from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytensor as pt
import arviz as az


"""
TLP_functions.py are in the VICKP project.

 - Input also the week of the year

"""

df_passengers = load_passengers_data()  # Load the data


training_split = int(len(df_passengers)*0.4)
 
normalization = 1000

x_train = df_passengers.index[:training_split].values/normalization
y_train = df_passengers['Passengers'].iloc[:training_split]/normalization


x_test = df_passengers.index[training_split:].values/normalization
y_test = df_passengers['Passengers'].iloc[training_split:]/normalization


#%% The trend part of prophet

n_changepoints = 10

t = np.arange(len(x_train))/len(x_train)
s = np.linspace(0, max(t), n_changepoints+2)[1:-1]
A = (t[:, None] > s)*1.

with pm.Model() as model:
    
    k = pm.Normal('k', mu = 0,sigma = 1)
    m = pm.Normal('m', mu = 0, sigma = 5)
    delta = pm.Laplace('delta', mu = 0, b = 1, shape = n_changepoints) # the b parameter has a big impact on the ability to over/underfit
    
    gamma = -s*delta
    
    growth = k+pm.math.dot(A,delta)
    offset = m+pm.math.dot(A,gamma)
    
    trend = growth*t+offset
    error = pm.HalfCauchy('sigma', 0.05)
    
    pm.Normal(
        'obs',
        mu = trend,
        sigma= error,
        observed = y_train
        )


with model:
    trace = pm.sample(tune=100, draws=500, chains=2, cores = 1)

#%% 
pm.plot_trace(trace)

# %% plot the trend

m1 = trace.posterior['m'].mean(('chain','draw')).values
k1 = trace.posterior['k'].mean(('chain','draw')).values
delta1 = trace.posterior['delta'].mean(('chain','draw')).values
growth1 = k1+np.dot(A, delta1)
offset1 = m1+np.dot(A,-s*delta1)
trend1 = growth1*t+offset1


plt.figure()
plt.plot(growth1, label = 'growth')
plt.plot(offset1, label = 'offset')
plt.plot(trend1, label = 'trend')
plt.plot(y_train, label = "data")
plt.legend()