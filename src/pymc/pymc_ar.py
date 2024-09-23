import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt

# Step 1: Create a Mock Time Series
np.random.seed(42)
n = 200  # Number of time points
t = np.arange(n)
data = (np.sin(t * 0.1) + np.random.normal(scale=0.4, size=n))  # Sine wave with noise

# Convert to a DataFrame
df = pd.DataFrame({'time': t, 'value': data})
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

coef_size = 10
LAM = 0.1 
init_size = 1
target_std = 0.1
rho_mu = 0
rho_sigma = 10
init_mu = 0
init_sigma = 100


prediction_length = len(x_test)
with pm.Model() as AR_model1:
    AR_model1.add_coord("obs_id", x_train)

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

with AR_model1:
    AR_model1.add_coords({"obs_id_fut_1": range(y_train.shape[0] - 1,y_train.shape[0] - 1+ prediction_length+1, 1)})
    AR_model1.add_coords({"obs_id_fut": range(y_train.shape[0],y_train.shape[0]+prediction_length, 1)})
    ar1_fut = pm.AR(
        "ar1_fut",
        init_dist=pm.DiracDelta.dist(ar1[..., -1]),
        rho=rho,
        tau=tau,
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
plt.ylim([0,max(y_test)*1.1])
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Forecasting with PyMC')
