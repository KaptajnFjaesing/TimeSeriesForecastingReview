import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic monthly data
np.random.seed(41)
T = 52
intervals = np.array(range(1, T+1))  # Months from 1 (January) to 12 (December)
years = 5  # Number of years of data
data = []

# Create a sinusoidal pattern with some noise
rate = 10 + 5 * np.sin(2 * np.pi * intervals / T)
for year in range(years):
    data.extend(np.random.poisson(lam=rate, size=intervals.shape))


data_s_training = np.array(data)
intervals_repeated = np.tile(intervals, years)

plt.figure()
plt.plot(intervals,rate)
plt.plot(intervals_repeated,data, 'o')
plt.xlabel('Week')
plt.ylabel('Value')


#%%


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
plt.xlabel('Month')
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
        x = pm.Data("x", data_x_training, dims=['obs_dim', 'coeff_dim'])
        beta = pm.Normal('beta', mu=0, sigma=1, dims = 'coeff_dim')
        #mu = pm.math.maximum(pm.math.dot(x, beta),0.001)
        mu = pm.math.exp(pm.math.dot(x, beta))
        y = pm.Poisson('y', mu = mu, observed = data_s_training, dims='obs_dim')
        
    return model

model = construct_pymc_model(
    data_x_training = data_x_training,
    data_s_training = data_s_training
    )

with model:
    posterior = pm.sample(tune=1000, draws=5000, chains=1)


# %%

# Create a high-resolution time axis for plotting, including a wrap-around point
high_res_months = np.linspace(1, T+1, 1000)  # Go slightly beyond 12 to ensure smooth wrapping

# Evaluate the model at high-resolution time points
data_x_test = fourier_series(high_res_months, n_terms, T)

chain = 0
beta_draw = posterior.posterior['beta'].mean((('chain'))).values

#preds = np.einsum("ij,kj->ik",data_x_test,beta_draw)

preds = np.exp(np.einsum("ij,kj->ik",data_x_test,beta_draw))


# Regular Cartesian Plot
plt.figure(figsize=(10, 6))
plt.scatter(intervals_repeated, data_s_training, color='blue', label='Data', alpha=0.6)

for pred in preds.T:
    plt.plot(high_res_months, pred, color='red', alpha=0.01)  # Set alpha to control opacity

plt.plot(high_res_months, preds.mean(axis = 1), color='black', label='Fitted Model Bayesian', linewidth=2)
plt.plot(intervals,rate)
plt.plot(high_res_months, y_pred, color='green', label='Fitted Model Frequentist', linewidth=2)
plt.xticks(ticks=np.arange(1, T+1), labels=intervals)
#plt.xlim(1, 12)
plt.xlabel('week')
plt.ylabel('Value')
plt.title('Regular Plot of Time Series with Fourier Series Fit')
plt.legend()
plt.grid(True)

# %%
# Polar Plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')
ax.scatter(theta, data, color='blue', label='Data', alpha=0.6)
for pred in preds.T:
    plt.plot(theta_high_res, pred, color='red', alpha=0.01)  # Set alpha to control opacity
ax.plot(theta_high_res, preds.mean(axis = 1), color='black', label='Fitted Model Bayesian', linewidth=2)
ax.set_theta_direction(-1)  # Optional: Reverse the direction of the plot (clockwise)
ax.set_theta_offset(np.pi / 2.0)  # Optional: Set 12 o'clock (January) as the top of the circle
ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_title('Polar Plot of Time Series with Continuous and Smooth Fourier Series Model', va='bottom')
plt.legend(loc='upper right')

# %%
with model:
    pm.set_data({"x": data_x_training})
    posterior_predictive = pm.sample_posterior_predictive(posterior)
    y_pred2 = posterior_predictive.posterior_predictive['y'].mean((('draw','chain'))).values

print()
print(data_s_training.mean())
print(y_pred2.mean())




#%%

import numpy as np
import matplotlib.pyplot as plt

# Define the time range
t = np.linspace(0, 2 * np.pi, 1000)  # 1000 points from 0 to 2*pi

# Define the sine wave
sine_wave = np.sin(t)

# Define the exponential of the sine wave
exp_sine_wave = np.exp(sine_wave)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot the sine wave
plt.subplot(1, 2, 1)
plt.plot(t, sine_wave, label='Sine Wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sine Wave')
plt.grid(True)
plt.legend()

# Plot the exponential of the sine wave
plt.subplot(1, 2, 2)
plt.plot(t, exp_sine_wave, label='Exp(Sine Wave)', color='orange')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Exponential of Sine Wave')
plt.grid(True)
plt.legend()


#%% Test for prior

beta = np.ones(data_x_training.shape[1])
ko = np.exp(np.dot(data_x_training,beta))

plt.figure()
plt.hist(ko)