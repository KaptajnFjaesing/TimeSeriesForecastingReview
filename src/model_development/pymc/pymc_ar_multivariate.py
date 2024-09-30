import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt

# Step 1: Create a Mock Time Series
np.random.seed(42)
n = 200  # Number of time points
t = np.arange(n)
data1 = (t * 0.1 + np.random.normal(scale=0.4, size=n))  # Sine wave with noise
data2 = (t * 0.4 + np.random.normal(scale=0.2, size=n))  # Sine wave with noise

# Convert to a DataFrame
df = pd.DataFrame({'time': t, 'value1': data1, 'value2': data2})
scale = 20

split = int(len(df)*0.77)
x_train = df['time'].values[:split]
y_train = df[['value1','value2']].values[:split]/scale

x_test = df['time'].values[split:]
y_test = df[['value1','value2']].values[split:]/scale

plt.figure(figsize=(12, 6))
plt.plot(x_train,y_train, label = "training data")
plt.plot(x_test,y_test, label = "test data")

#%% Model 1

coef_size = 10
LAM = 0.1 
target_std = 0.1
rho_mu = 0
rho_sigma = 10
init_mu = 0
init_sigma = 100


coords = {
    "number_of_time_series": range(y_train.shape[1]),
    "coef_size": range(coef_size+1),
    "number_of_observations": range(x_train.shape[0])
    }

prediction_length = len(x_test)
with pm.Model(coords = coords) as AR_model1:
    AR_model1.add_coord("obs_id", x_train)
    y1 = pm.Data("y", y_train)

    rho = pm.Normal("coefs",
                      mu = rho_mu,
                      sigma= rho_sigma,
                      shape = (2,coef_size+1)
                      )
    tau = pm.Exponential("tau", lam=LAM)
    init = pm.Normal.dist(
        mu = init_mu,
        sigma = init_sigma,
        shape = (2,coef_size+1)
    )
    ar1 = pm.AR(
        "ar",
        rho = rho,
        tau = tau,
        init_dist = init,
        constant=True,
        dims = ['number_of_time_series', 'number_of_observations']
    ).T
    pm.Normal(
        "target_distribution",
        mu=ar1,
        sigma=target_std,
        observed=y1,
        dims=["number_of_observations",'number_of_time_series']
        )
    trace = pm.sample(tune=100, draws=200, chains=1, return_inferencedata=True)


#%%

with pm.Model(coords = coords) as ar_test_model:
    ar_test_model.add_coords({"obs_id_fut_1": range(y_train.shape[0] - 1,y_train.shape[0] - 1+ prediction_length+1, 1)})
    ar_test_model.add_coords({"obs_id_fut": range(y_train.shape[0],y_train.shape[0]+prediction_length, 1)})
    
    rho = pm.Normal("coefs",
                      mu = rho_mu,
                      sigma= rho_sigma,
                      shape = (2,coef_size+1)
                      )

    initial_distribution = pm.DiracDelta.dist(
        ar1[-coef_size:, :].T,  
        shape=(2,coef_size+1)
    )
    
    ar1_fut = pm.AR(
        "ar1_fut",
        init_dist = initial_distribution,
        rho=rho,
        constant = True,
        dims = ['number_of_time_series', 'obs_id_fut_1']
    ).T

    yhat_fut = pm.Normal(
        "yhat_fut",
        mu=ar1_fut[1:,:],
        sigma=target_std,
        dims=["obs_id_fut",'number_of_time_series']
        )
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names = ["yhat_fut"], predictions = True )

#%%

mean_predictions_test = posterior_predictive.predictions['yhat_fut'].mean(("chain", "draw")).values


plt.figure(figsize=(12, 6))
plt.plot(x_train,y_train, label = "training data")
plt.plot(x_test,y_test, label = "test data")
plt.plot(x_test,mean_predictions_test, label='Forecast AR model 1', linestyle='--')
plt.ylim([0,5])
