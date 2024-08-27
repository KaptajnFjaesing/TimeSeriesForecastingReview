from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pytensor as pt

df_passengers = load_passengers_data()  # Load the data

training_split = int(len(df_passengers)*0.7)
 
x_train_unnormalized = df_passengers.index[:training_split].values
y_train_unnormalized = df_passengers['Passengers'].iloc[:training_split].values

x_train_mean = x_train_unnormalized.mean()
x_train_std = x_train_unnormalized.std()

y_train_mean = y_train_unnormalized.mean()
y_train_std = y_train_unnormalized.std()

x_train = ((x_train_unnormalized-x_train_mean)/x_train_std).reshape(-1,1)/10
y_train = ((y_train_unnormalized-y_train_mean)/y_train_std).reshape(-1,1)/10

x_test = ((df_passengers.index[training_split:].values-x_train_mean)/x_train_std).reshape(-1,1)/10
y_test = ((df_passengers['Passengers'].iloc[training_split:].values-y_train_mean)/y_train_std).reshape(-1,1)/10

# %%

"""
The approach here is to try a 1-1 map between input and output.

It does not work well. The reason is that you need a larger context.


"""

import src.pymc_functions as fu_pymc

activation = 'relu'
n_hidden_layer1 = 20
n_hidden_layer2 = 20


model = fu_pymc.construct_pymc_tlp(
    x_train = x_train,
    y_train = y_train,
    n_hidden_layer1 = n_hidden_layer1,
    n_hidden_layer2 = n_hidden_layer2,
    activation = activation
    )


draws = 200
with model:
    trace = pm.sample(
        tune = 100,
        draws = draws,
        chains = 1,
        #target_accept = 0.9,
        return_inferencedata = True
        )
    
# %%


X = x_train

with model:
    pm.set_data({'x':X})
    posterior_predictive = pm.sample_posterior_predictive(
        trace = trace,
        predictions = True
        )


#%%
import arviz as az

preds_out_of_sample = posterior_predictive.predictions_constant_data['x']
model_preds = posterior_predictive.predictions

plt.figure()
plt.plot(
    preds_out_of_sample,
    model_preds["y_obs"].mean(("chain", "draw"))
    )
hdi_values = az.hdi(model_preds)["y_obs"].transpose("hdi", ...)

plt.fill_between(
    preds_out_of_sample.values[:,0],
    hdi_values[0].values[:,0],  # lower bound of the HDI
    hdi_values[1].values[:,0],  # upper bound of the HDI
    color="gray",   # color of the shaded region
    alpha=0.4,      # transparency level of the shaded region
)
plt.plot(x_train,y_train)
plt.plot(x_test,y_test, color = "blue")


