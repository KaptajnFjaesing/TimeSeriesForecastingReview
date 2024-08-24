#%%
from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
TLP_functions.py are in the VICKP project.

 - Input also the week of the year

"""

df_passengers = load_passengers_data()  # Load the data



training_split = int(len(df_passengers)*0.7)

x_train = df_passengers.index[:training_split].values
y_train = df_passengers['Passengers'].iloc[:training_split]


x_test = df_passengers.index[training_split:].values
y_test = df_passengers['Passengers'].iloc[training_split:]


#%%

np.random.seed(42)
x_train = np.linspace(0, 10, 100)[:, None]  # 100 time points
y_train = np.sin(x_train).flatten() + 0.5 * np.random.randn(100)  # Signal + noise


def construct_pymc_model(
        x_train: np.array,
        y_train: np.array
        ):

    with pm.Model() as model:
        # Define the covariance function
        ℓ = pm.Gamma("ℓ", alpha=2, beta=1)
        η = pm.HalfCauchy("η", beta=5)
    
        # Covariance function
        cov_func = η ** 2 * pm.gp.cov.ExpQuad(1, ℓ)
    
        # Gaussian Process definition
        gp = pm.gp.Marginal(cov_func=cov_func)
    
        # Define the GP likelihood
        σ = pm.HalfCauchy("σ", beta=5)
        y_obs = gp.marginal_likelihood("y_obs", X= x_train, y=y_train, sigma=σ)

        return model, gp


model, gp = construct_pymc_model(x_train = x_train, y_train = y_train)

with model:
    posterior = pm.sample(tune=50, draws=100, chains=1, return_inferencedata=False)

# %%
x_test = np.linspace(10, 15, 50)[:, None]  # New time points for prediction

with model:
    mu, var = gp.predict(x_test, point=posterior[-1], diag=True, pred_noise=True)

# Convert variance to standard deviation for plotting
sd = np.sqrt(var)

# %%

plt.figure(figsize=(12, 6))
plt.plot(x_train, y_train, "b.", label="Observed data")
plt.plot(x_test, mu, "r", label="Predicted mean")
plt.fill_between(x_test.flatten(), mu - 2 * sd, mu + 2 * sd, color="red", alpha=0.3, label="95% confidence interval")
plt.legend()
plt.show()

# %% 


import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Assuming X, y, and the model are already defined

np.random.seed(42)
X = np.linspace(0, 10, 100)[:, None]  # 100 time points
y = np.sin(X).flatten() + 0.5 * np.random.randn(100)  # Signal + noise


# Define how many steps into the future we want to predict
M = 20  # For example, 20 steps into the futurel
X_new = np.linspace(X[-1], X[-1] + M, M + 1)[:, None]  # Time points for prediction

# Store predictions
mu_list = []
var_list = []

with model:
    # Extract the last sample from the trace
    eta_sample = trace['η'][-1]
    sigma_sample = trace['σ'][-1]
    ell_sample = trace['ℓ'][-1]

    # Create the point dictionary
    point = {
        'η': eta_sample,
        'σ': sigma_sample,
        'ℓ': ell_sample
    }

    # Start with the last observed point
    last_x = X[-1].reshape(1, -1)
    
    for i in range(M):
        # Predict the next step
        mu, var = gp.predict(last_x, point=point, diag=True, pred_noise=True)

        # Append results
        mu_list.append(mu)
        var_list.append(var)

        # Use the predicted mean as the new input for the next prediction
        last_x = np.array([[X[-1] + i + 1]])

# Convert lists to arrays
mu = np.concatenate(mu_list)
var = np.concatenate(var_list)

# Convert variance to standard deviation for plotting
sd = np.sqrt(var)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(X, y, "b.", label="Observed data")
plt.plot(X_new.flatten(), mu, "r", label=f"Predicted mean ({M} steps)")
plt.fill_between(X_new.flatten(), mu - 2 * sd, mu + 2 * sd, color="red", alpha=0.3, label="95% confidence interval")
plt.legend()
plt.show()
