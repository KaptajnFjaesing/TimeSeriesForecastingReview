#%%
from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

"""
TLP_functions.py are in the VICKP project.

 - Input also the week of the year

"""

df_passengers = load_passengers_data()  # Load the data


y_normalization = 1000

training_split = int(len(df_passengers)*0.7)
n_test = len(df_passengers)-training_split

x_train = df_passengers.index[:training_split].values.reshape(training_split,1)/y_normalization
y_train = df_passengers['Passengers'].iloc[:training_split].values.reshape(training_split,)/y_normalization

print(x_train.shape)
print(y_train.shape)

x_test = df_passengers.index[training_split:].values.reshape(n_test,1)/y_normalization
y_test = df_passengers['Passengers'].iloc[training_split:].values.reshape(n_test,)/y_normalization

print(x_test.shape)
print(y_test.shape)
#%%

#np.random.seed(42)
x_train2 = np.linspace(0, 10, 100)[:, None]  # 100 time points
y_train2 = np.sin(x_train2).flatten() + 0.5 * np.random.randn(100)  # Signal + noise

print(x_train2.shape)
print(y_train2.shape)

def construct_pymc_model(
        x_train: np.array,
        y_train: np.array
        ):

    with pm.Model() as model:
        # Define the covariance function
        xi = pm.Gamma("xi", alpha=2, beta=1)
        eta = pm.HalfCauchy("eta", beta=5)
    
        # Covariance function
        cov_func = eta ** 2 * pm.gp.cov.ExpQuad(1, xi)
    
        # Gaussian Process definition
        gp = pm.gp.Marginal(cov_func=cov_func)
    
        # Define the GP likelihood
        sigma = pm.HalfCauchy("sigma", beta=5)
        y_obs = gp.marginal_likelihood("y_obs", X= x_train, y=y_train, sigma=sigma)

        return model, gp

model, gp = construct_pymc_model(x_train = x_train, y_train = y_train)

with model:
    posterior = pm.sample(
        tune=500,
        draws=1000,
        chains=1,
        return_inferencedata=True
        )

#%%
from tqdm import tqdm

n_samples = posterior.posterior.dims["draw"]
mu_preds = np.zeros((n_samples, len(x_test)))
var_preds = np.zeros((n_samples, len(x_test)))

# Loop over each posterior sample
for i in tqdm(range(n_samples)):
    # Extract current set of parameters
    xi_sample = posterior.posterior["xi"].mean('chain').isel(draw=i).values
    eta_sample = posterior.posterior["eta"].mean('chain').isel(draw=i).values
    sigma_sample = posterior.posterior["sigma"].mean('chain').isel(draw=i).values

    # Reconstruct the covariance function with current parameters
    #cov_func = eta_sample**2 * pm.gp.cov.ExpQuad(1, xi_sample)

    # Make predictions
    with model:
        mu, var = gp.predict(
            x_test,
            point={"xi": xi_sample, "eta": eta_sample, "sigma": sigma_sample},
            diag=True,
            pred_noise=True
        )
    
    # Store the predictions
    mu_preds[i, :] = mu
    var_preds[i, :] = var


# %%

plt.figure(figsize=(12, 6))
plt.plot(x_train, y_train, "b.", label="Observed data")
for pred in mu_preds:
    plt.plot(x_test, pred, "r", alpha = 0.02)
plt.plot(x_test, mu_preds.mean(axis = 0), color = "black", label="Predicted mean")
plt.legend()

