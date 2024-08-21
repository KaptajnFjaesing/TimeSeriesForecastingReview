# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:24:00 2024

@author: roman
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic monthly data
np.random.seed(42)
T = 52
intervals = np.array(range(1, T+1))  # Months from 1 (January) to 12 (December)
years = 5  # Number of years of data
data = []

# Create a sinusoidal pattern with some noise
rate = 10 + 5 * np.sin(2 * np.pi * intervals / T)
for year in range(years):
    noise = np.random.poisson(lam=rate, size=intervals.shape)
    data.extend(noise)

data_s_training = np.array(data)
intervals_repeated = np.tile(intervals, years)

# Define the symmetric Fourier series model
def fourier_series(t, n_terms, T):
    terms = [np.ones(t.shape)]  # a_0 term
    for k in range(1, n_terms + 1):
        terms.append(np.cos(2 * np.pi * k * t / T))
        terms.append(np.sin(2 * np.pi * k * t / T))
    return np.column_stack(terms)

# Number of terms in the Fourier series
n_terms = 5  # Adjust the number of terms for fitting

# Construct the design matrix for the Fourier series using the original data
# It has dimensions (data, coefficients) where the latter is 1+2*n_terms
data_x_training = fourier_series(intervals_repeated, n_terms, T)

# Perform the linear regression manually (least squares solution)
beta = np.linalg.inv(data_x_training.T @ data_x_training) @ data_x_training.T @ data

# Create a high-resolution time axis for plotting, including a wrap-around point
high_res_months = np.linspace(1, T+1, 1000)  # Go slightly beyond 12 to ensure smooth wrapping

# Evaluate the model at high-resolution time points
data_x_test = fourier_series(high_res_months, n_terms, T)
y_pred = data_x_test @ beta

# Regular Cartesian Plot
plt.figure(figsize=(10, 6))
plt.scatter(intervals_repeated, data_s_training, color='blue', label='Data', alpha=0.6)
plt.plot(high_res_months, y_pred, color='red', label='Fitted Model', linewidth=2)
plt.plot(intervals,rate, label = "True rate")
plt.xticks(ticks=np.arange(1, T+1), labels=intervals)
#plt.xlim(1, 12)
plt.xlabel('Week')
plt.ylabel('Value')
plt.title('Regular Plot of Time Series with Fourier Series Fit')
plt.legend()
plt.grid(True)


# Convert months to angles (theta) for polar plot
theta = 2 * np.pi * (intervals_repeated - 1) / T
theta_high_res = 2 * np.pi * (high_res_months - 1) / T

# Polar Plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')
ax.scatter(theta, data, color='blue', label='Data', alpha=0.6)
ax.plot(theta_high_res, y_pred, color='red', label='Fitted Model', linewidth=2)
ax.set_theta_direction(-1)  # Optional: Reverse the direction of the plot (clockwise)
ax.set_theta_offset(np.pi / 2.0)  # Optional: Set 12 o'clock (January) as the top of the circle
ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_title('Polar Plot of Time Series with Continuous and Smooth Fourier Series Model', va='bottom')
plt.legend(loc='upper right')

# %%


import pymc as pm
import numpy as np
import matplotlib.pyplot as plt


def construct_pymc_model(
        data_x_training: np.array,
        data_s_training: np.array
        ):

    with pm.Model() as model:
        
        n_features = data_x_training.shape[1]
        
        x = pm.Data("x", data_x_training, dims=['obs_dim', 'coeff_dim'])
        
        precision_a = pm.Gamma('precision_a', alpha=30, beta=0.1)
        precision_b = pm.Gamma('precision_b', alpha=30, beta=0.1)
        param_a = pm.Normal('param_a', 0, sigma=1/np.sqrt(precision_a), shape=(n_features,))
        param_b = pm.Normal('param_b', 0, sigma=1/np.sqrt(precision_b))
        mu = pm.Deterministic('T', pm.math.exp(param_b + pm.math.dot(x, param_a)))
        y = pm.Poisson('y', mu = mu, observed = data_s_training, dims='obs_dim')
        
    return model

model = construct_pymc_model(
    data_x_training = data_x_training,
    data_s_training = data_s_training
    )

with model:
    posterior = pm.sample(tune=500, draws=1000, chains=1)


# %%

# Create a high-resolution time axis for plotting, including a wrap-around point
high_res_months = np.linspace(1, T+1, 1000)  # Go slightly beyond 12 to ensure smooth wrapping

# Evaluate the model at high-resolution time points
data_x_test = fourier_series(high_res_months, n_terms, T)

chain = 0
param_a = posterior.posterior['param_a'].mean((('chain'))).values
param_b = posterior.posterior['param_b'].mean((('chain'))).values

plt.figure()
plt.hist(param_b, density= True)
plt.hist(param_a[0], density = True)



preds = np.exp(param_b+np.einsum("ij,kj->ik",data_x_test,param_a))


# Regular Cartesian Plot
plt.figure(figsize=(10, 6))
plt.scatter(intervals_repeated, data_s_training, color='blue', label='Data', alpha=0.6)

plt.plot(high_res_months, preds.mean(axis = 1), color='black', label='Fitted Model Bayesian', linewidth=2)
plt.plot(intervals,rate)
plt.xticks(ticks=np.arange(1, T+1), labels=intervals)
#plt.xlim(1, 12)
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Regular Plot of Time Series with Fourier Series Fit')
plt.legend()
plt.grid(True)

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Parameters for the Gamma distribution
alpha = 30  # Shape parameter
beta = 0.1     # Scale parameter

# Create an array of x values
x = np.linspace(0, 20, 1000)

# Compute the probability density function (PDF) of the Gamma distribution
pdf = gamma.pdf(x, a=alpha, scale=beta)

# Plot the Gamma distribution
plt.figure(figsize=(8, 6))
plt.plot(x, pdf, label=f'Gamma Distribution\nα={alpha}, β={beta}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Gamma Distribution')
plt.grid(True)
plt.legend()
plt.show()
