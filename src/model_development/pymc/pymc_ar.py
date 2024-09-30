import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt

def generate_ar_time_series(n_timesteps, coefs, noise_std, initial_values):
    """
    Generate an AR(2) time series with specified coefficients and noise.
    
    Args:
    n_timesteps (int): Number of timesteps to generate.
    coefs (list of floats): AR coefficients [phi1, phi2].
    noise_std (float): Standard deviation of Gaussian noise.
    initial_values (list of floats): Initial values to start the time series.
    
    Returns:
    np.ndarray: Generated AR(2) time series.
    """
    # Pre-allocate time series array
    time_series = np.zeros(n_timesteps)
    
    # Set initial values
    time_series[0:2] = initial_values
    
    # Generate the AR(2) series
    for t in range(2, n_timesteps):
        noise = np.random.normal(0, noise_std)
        time_series[t] = coefs[0] * time_series[t-1] + coefs[1] * time_series[t-2] + noise
    
    return time_series

# Parameters for AR(2) model
n_timesteps = 200
coefs = [0.6, -0.3]  # AR coefficients (phi1, phi2)
noise_std = 0.1      # Noise standard deviation
initial_values = [0.5, 0.3]  # Initial values for the first two timesteps

# Generate two time series
ar_series_1 = generate_ar_time_series(n_timesteps, coefs, noise_std, initial_values)
ar_series_2 = generate_ar_time_series(n_timesteps, coefs, noise_std, initial_values)

df = pd.DataFrame({'time': np.arange(n_timesteps), 'value': ar_series_1})
scale = 20

split = int(len(df)*0.77)
x_train = df['time'].values[:split]
y_train = df['value'].values[:split]/scale

x_test = df['time'].values[split:]
y_test = df['value'].values[split:]/scale

plt.figure(figsize=(12, 6))
plt.plot(x_train,y_train, label = "training data")
plt.plot(x_test,y_test, label = "test data")

#%% Model 1

"""
Notes:
    - Model 1 seem to correctly predict the offset, which model 2 cannot seem to do.
    - LAM is a very sensitive parameter together with target_std for fitting a linear model.
        A too large LAM will make the predictions explode.
        A too large target_std will make the predictions off.
        
        It seems that LAM and target std to some degree counteract each other,
        so one can be adjusted up and the other down, yielding approximately
        an unchanged result.
        
    - What is init_size?
    
    

"""

coef_size = 2
target_std = 0.2
rho_mu = 0
rho_sigma = 1
init_mu = 0
init_sigma = 1


prediction_length = len(x_test)
with pm.Model() as AR_model1:
    AR_model1.add_coord("obs_id", x_train)

    y1 = pm.Data("y", y_train, dims="obs_id2")
    rho = pm.Normal("coefs",
                      mu = rho_mu,
                      sigma= rho_sigma,
                      shape = coef_size
                      )
    init = pm.Normal.dist(
        mu = init_mu,
        sigma = init_sigma,
        shape = coef_size-1
    )
    ar1 = pm.AR(
        "ar",
        rho = rho,
        sigma = 1,
        init_dist=pm.Normal.dist(0,10),
        constant=True,
        dims="obs_id",
    )
    target_distribution = pm.Normal("target_distribution", mu=ar1, sigma=target_std, observed=y1, dims="obs_id")
    trace = pm.sample(tune=100, draws=500, chains=1, return_inferencedata=True)

with AR_model1:
    AR_model1.add_coords({"obs_id_fut_1": range(y_train.shape[0] - 1,y_train.shape[0] - 1+ prediction_length+1, 1)})
    AR_model1.add_coords({"obs_id_fut": range(y_train.shape[0],y_train.shape[0]+prediction_length, 1)})
    ar1_fut = pm.AR(
        "ar1_fut",
        init_dist=pm.DiracDelta.dist(ar1[..., -1]),
        rho=rho,
        sigma = 1,
        constant=True,
        dims="obs_id_fut_1",
    )
    yhat_fut = pm.Normal("yhat_fut", mu=ar1_fut[1:], sigma=target_std, dims="obs_id_fut")
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["target_distribution", "yhat_fut"], predictions = True )


mean_predictions_training = posterior_predictive.predictions['target_distribution'].mean(("chain", "draw")).values
mean_predictions_test = posterior_predictive.predictions['yhat_fut'].mean(("chain", "draw")).values


plt.figure(figsize=(12, 6))
plt.plot(x_train,y_train, label = "training data")
plt.plot(x_test,y_test, label = "test data")
plt.plot(x_test,mean_predictions_test, label='Forecast AR model 1', linestyle='--')

# %% Model 2

"""
This model cannot feel the last datapoints, which is why its predictions equal
the N first data, for a forecast of N.

"""

init_size = 1
rho_sigma = 0.1
init_sigma = 1

with pm.Model() as AR_model2:
    t1 = pm.Data("t", x_train, dims="obs_id")
    y1 = pm.Data("y", y_train, dims="obs_id2")
    rho = pm.Normal("coefs",
                      mu = rho_mu,
                      sigma= rho_sigma,
                      shape = coef_size+1
                      )
    tau = pm.Exponential("tau", lam=LAM)
    init = pm.Normal.dist(
        mu = init_mu,
        sigma = init_sigma,
        shape = init_size
    )
    ar1 = pm.AR(
        "ar",
        rho = rho,
        tau=tau,
        init_dist=init,
        constant=True,
        steps = t1.shape[0]-coef_size,
        dims="obs_id",
    )
    target_distribution = pm.Normal("target_distribution", mu=ar1, sigma=target_std, observed=y1, dims="obs_id")
    trace = pm.sample(tune=100, draws=500, chains=1, return_inferencedata=True)

with AR_model2:
    pm.set_data({'t': x_test})
    posterior_predictive2 = pm.sample_posterior_predictive(trace, predictions = True )

mean_predictions_test2 = posterior_predictive2.predictions['target_distribution'].mean(("chain", "draw")).values


#%%

plt.figure(figsize=(12, 6))
plt.plot(x_train,y_train, label = "training data")
plt.plot(x_test,y_test, label = "test data")
plt.plot(x_test,mean_predictions_test, label='Forecast AR model 1', linestyle='--')
plt.plot(x_test,mean_predictions_test2, label='Forecast AR model 2', linestyle='--')
plt.ylim([-0.2,0.2])
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Forecasting with PyMC')
