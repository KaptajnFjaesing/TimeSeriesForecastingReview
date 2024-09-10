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


training_split = int(len(df_passengers)*0.7)
 
normalization = 100

x_train = df_passengers.index[:training_split].values/normalization
y_train = df_passengers['Passengers'].iloc[:training_split]/normalization


x_test = df_passengers.index[training_split:].values/normalization
y_test = df_passengers['Passengers'].iloc[training_split:]/normalization


# %% trend and seasonality

def create_fourier_features(
        t: np.array,
        n_fourier_components: int,
        seasonality_period: float = 365.25
        ):
    x = 2 * np.pi * (np.arange(n_fourier_components)+1) * t[:, None] / seasonality_period
    return pm.math.concatenate((np.cos(x), np.sin(x)), axis = 1)

def pymc_prophet(
        x_train: np.array,
        y_train: np.array,
        seasonality_period: float,
        n_changepoints: int = 10,
        n_fourier_components: int = 10
        ):
    
    #s = pt.tensor.arange(0, pm.math.max(x_train), n_changepoints+2)[1:-1]
    #A = (x_train[:, None] > s)*1.
    
    with pm.Model() as model:
        x = pm.Data('x', x_train, dims = ['n_obs'])
        y = pm.Data('y', y_train)
        
        s = pt.tensor.linspace(0, pm.math.max(x), n_changepoints+2)[1:-1]
        A = (x[:, None] > s)*1.
        
        k = pm.Normal('k', mu = 0, sigma = 5)
        m = pm.Normal('m', mu = 0, sigma = 5)
        delta = pm.Laplace('delta', mu = 0, b = 0.5, shape = n_changepoints) # the b parameter has a big impact on the ability to over/underfit
        gamma = -s*delta
        growth = k+pm.math.dot(A,delta)
        offset = m+pm.math.dot(A,gamma)
        trend = growth*x+offset
        fourier_coefficients = pm.Normal('beta', mu = 0, sigma = 5, shape = 2*n_fourier_components) # parameters for each sine and cosine
        
        fourier_sine_terms = create_fourier_features(
            t = x,
            n_fourier_components = n_fourier_components,
            seasonality_period = seasonality_period
            )
        seasonality_scale = pm.Normal('ss', mu = 0, sigma = 5)
        seasonality = pm.math.dot(fourier_sine_terms,fourier_coefficients)
        prediction = trend + seasonality*(1+x*seasonality_scale)
        error = pm.HalfCauchy('sigma', 0.5)
        pm.Normal(
            'obs',
            mu = prediction,
            sigma= error,
            observed = y,
            dims = ['n_obs']
            )
    return model
    
n_changepoints = 10
n_fourier_components = 5

model = pymc_prophet(
    x_train = x_train,
    y_train = y_train,
    seasonality_period = 12/normalization,
    n_changepoints = n_changepoints,
    n_fourier_components = n_fourier_components
    )
    
with model:
    trace = pm.sample(tune=500, draws=2000, chains=1, cores = 1)

#%% 
pm.plot_trace(trace)

# %% Understand the model in order to tune it
s = np.linspace(0, max(x_train), n_changepoints+2)[1:-1]
A = (x_train[:, None] > s)*1.
x = 2 * np.pi * (np.arange(n_fourier_components)+1) * x_train[:, None] / (13/normalization)
hest = np.concatenate((np.cos(x), np.sin(x)), axis = 1)

m1 = trace.posterior['m'].mean(('chain','draw')).values
k1 = trace.posterior['k'].mean(('chain','draw')).values
delta1 = trace.posterior['delta'].mean(('chain','draw')).values
beta1 = trace.posterior['beta'].mean(('chain','draw')).values
growth1 = k1+np.dot(A, delta1)
offset1 = m1+np.dot(A,-s*delta1)
trend1 = growth1*x_train+offset1

seasonality1 = np.dot(hest,beta1)


plt.figure()
#plt.plot(growth1, label = 'growth')
#plt.plot(offset1, label = 'offset')
plt.plot(trend1, label = 'trend')
plt.plot(seasonality1, label = 'seasonality')
plt.plot(trend1+seasonality1, label = 'trend and seasonality')
plt.plot(y_train, label = "data")
plt.legend()

#%%

with model:
    pm.set_data({'x':x_test})
    posterior_predictive = pm.sample_posterior_predictive(trace, predictions=True)

predictions = posterior_predictive.predictions['obs'].mean((('chain'))).values

plt.figure()
plt.plot(x_test,predictions.mean(axis = 0), color = 'black')
for i in predictions:
    plt.plot(x_test,i, alpha = 0.01, color = 'red')
plt.plot(x_train,y_train)
plt.plot(x_test,y_test, color = "blue")


