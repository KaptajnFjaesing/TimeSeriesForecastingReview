from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pytensor as pt

df_passengers = load_passengers_data()  # Load the data

training_split = int(len(df_passengers)*0.7)
 
x_train_unnormalized = df_passengers.index[:training_split].values
y_train_unnormalized = df_passengers['Passengers'].iloc[:training_split]

x_train_mean = x_train_unnormalized.mean()
x_train_std = x_train_unnormalized.std()

y_train_mean = y_train_unnormalized.mean()
y_train_std = y_train_unnormalized.std()

x_train = (x_train_unnormalized-x_train_mean)/x_train_std
y_train = (y_train_unnormalized-y_train_mean)/y_train_std

x_test = (df_passengers.index[training_split:].values-x_train_mean)/x_train_std
y_test = (df_passengers['Passengers'].iloc[training_split:].values-y_train_mean)/y_train_std

# %% trend and seasonality

def create_fourier_features(
        t: np.array,
        n_fourier_components: int,
        seasonality_period: float
        ):
    x = 2 * np.pi * (np.arange(n_fourier_components)+1) * t[:, None] / seasonality_period
    return pm.math.concatenate((np.cos(x), np.sin(x)), axis = 1)

def pymc_prophet(
        x_train: np.array,
        y_train: np.array,
        seasonality_period_baseline: float,
        n_changepoints: int = 10,
        n_fourier_components: int = 10
        ):

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
        fourier_coefficients_1 = pm.Normal('fourier_coefficients1', mu = 0, sigma = 5, shape = 2*n_fourier_components) # parameters for each sine and cosine
        fourier_coefficients_2 = pm.Normal('fourier_coefficients2', mu = 0, sigma = 5, shape = 2*n_fourier_components) # parameters for each sine and cosine
        season_parameter = pm.Normal('season_parameter', mu = 0, sigma = 1)
        seasonality_period = seasonality_period_baseline*pm.math.exp(season_parameter)
        
        fourier_sine_terms = create_fourier_features(
            t = x,
            n_fourier_components = n_fourier_components,
            seasonality_period = seasonality_period
            )
        seasonality_1 = pm.math.dot(fourier_sine_terms,fourier_coefficients_1)
        seasonality_2 = pm.math.dot(fourier_sine_terms,fourier_coefficients_2)

        prediction = trend*(1+seasonality_2) + seasonality_1
        
        error = pm.HalfCauchy('sigma', beta = 1)
        pm.Normal(
            'obs',
            mu = prediction,
            sigma= error,
            observed = y,
            dims = ['n_obs']
            )
    return model
    
n_changepoints = 20
n_fourier_components = 5
seasonality_period = (x_train[11]-x_train[0])/0.9


model = pymc_prophet(
    x_train = x_train,
    y_train = y_train,
    seasonality_period_baseline = seasonality_period,
    n_changepoints = n_changepoints,
    n_fourier_components = n_fourier_components
    )
    
with model:
    trace = pm.sample(
        tune = 500,
        draws = 2000, 
        chains = 1,
        cores = 1
        )

#%% 
pm.plot_trace(trace)

#%% Generate predictions

X = x_test

with model:
    pm.set_data({'x':X})
    posterior_predictive = pm.sample_posterior_predictive(trace = trace, predictions=True)

predictions = posterior_predictive.predictions['obs'].mean((('chain'))).values

plt.figure()
plt.plot(X,predictions.mean(axis = 0), color = 'black')
for i in predictions:
    plt.plot(X,i, alpha = 0.01, color = 'red')
plt.plot(x_train,y_train)
plt.plot(x_test,y_test, color = "blue")


# %% Alternative approach 

"""
This is an alternative approach recommended by the official documentation

https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html


"""
import arviz as az

preds_out_of_sample = posterior_predictive.predictions_constant_data.sortby('x')['x']
model_preds = posterior_predictive.predictions.sortby(preds_out_of_sample)

plt.figure()
plt.plot(
    preds_out_of_sample,
    model_preds["obs"].mean(("chain", "draw"))
    )
plt.vlines(
    preds_out_of_sample,
    *az.hdi(model_preds)["obs"].transpose("hdi", ...),
    alpha=0.8,
)
for i in predictions:
    plt.plot(X,i, alpha = 0.01, color = 'red')


# %% Understand the model in order to tune it
"""

outdated code

s = np.linspace(0, max(x_train), n_changepoints+2)[1:-1]
A = (x_train[:, None] > s)*1.
x = 2 * np.pi * (np.arange(n_fourier_components)+1) * x_train[:, None] / seasonality_period
hest = np.concatenate((np.cos(x), np.sin(x)), axis = 1)

m1 = trace.posterior['m'].mean(('chain','draw')).values
k1 = trace.posterior['k'].mean(('chain','draw')).values
delta1 = trace.posterior['delta'].mean(('chain','draw')).values
beta1 = trace.posterior['beta'].mean(('chain','draw')).values
#seasonality_scale1 = trace.posterior['ss'].mean(('chain','draw')).values


growth1 = k1+np.dot(A, delta1)
offset1 = m1+np.dot(A,-s*delta1)
trend1 = growth1*x_train+offset1

seasonality1 = np.dot(hest,beta1)

prediction1 = trend1*(1+seasonality2) + seasonality1


plt.figure()
plt.plot(trend1, label = 'trend')
plt.plot(seasonality1, label = 'seasonality')
plt.plot(prediction1, label = 'prediction')
plt.plot(y_train, label = "data")
plt.legend()

"""

