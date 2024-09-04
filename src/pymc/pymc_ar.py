import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt

# Step 1: Create a Mock Time Series
np.random.seed(42)
n = 200  # Number of time points
t = np.arange(n)
data = np.sin(t * 0.1) + np.random.normal(scale=0.5, size=n)  # Sine wave with noise

# Convert to a DataFrame
df = pd.DataFrame({'time': t, 'value': data})

# Prepare the data for PyMC
series = df['value'].values

# Step 2: Define the AR Model
with pm.Model() as model:
    ar_order = 3  # Define the order of the AR process
    
    # Define the coefficients for the AR process (including constant term)
    coefs = pm.Normal("coefs", mu=0, sigma=1, size=ar_order + 1)  # +1 for the constant term
    
    # Initial distribution for the AR process
    init_dist = pm.Normal.dist(mu=0, sigma=100, shape=ar_order)
    
    # Define the AR process
    ar_process = pm.AR("ar_process", 
                       rho=coefs, 
                       sigma=1.0, 
                       init_dist=init_dist, 
                       constant=True,
                       observed=series
                       )
with model:
    # Inference
    trace = pm.sample(tune=50, draws=100, chains=1, return_inferencedata=False)

#%%
# Step 3: Make Predictions
# Predict future values
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace)

#%%
# Extract the predictions
predictions = posterior_predictive.posterior_predictive["ar_process"].mean((('chain'))).values

# Calculate the mean of the predictions
mean_predictions = np.mean(predictions, axis=0)

# Step 4: Plot Results
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['value'], label='Actual Data')
plt.plot(df['time'][-len(mean_predictions):], mean_predictions, label='Predicted Data', linestyle='--')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Forecasting with PyMC')
plt.show()
