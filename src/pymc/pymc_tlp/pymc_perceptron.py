#%% Load data and feature engineering
from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_passengers = load_passengers_data()  # Load the data

def create_lagged_features(df, column_name, n_lags, n_ahead):
    """Create lagged features for a given column in the DataFrame, with future targets."""
    lagged_df = pd.DataFrame()
    
    # Create lagged features
    for i in range(n_lags, -1, -1):
        lagged_df[f'lag_{i}'] = df[column_name].shift(i)
    
    # Create future targets
    for i in range(1, n_ahead + 1):
        lagged_df[f'target_{i}'] = df[column_name].shift(-i)
    
    lagged_df.dropna(inplace=True)  # Remove rows with NaN values due to shifting
    return lagged_df

# Parameters
n_lags = 35  # Number of lagged time steps to use as features
forecast_horizon = 26  # Number of future time steps to predict

# Create lagged features and future targets
lagged_features = create_lagged_features(df_passengers, 'Passengers', n_lags = n_lags, n_ahead = forecast_horizon)

# Split the data
train_size = int(len(lagged_features) * 0.6)  # 60% for training, 40% for testing
training_data = lagged_features.iloc[:train_size]
test_data = lagged_features.iloc[train_size:]

normalization = 400
# Define features and targets
x_train = training_data[training_data.columns[:n_lags+1]]/normalization  # Lagged features
y_train = training_data[training_data.columns[n_lags+1:]]/normalization  # Future targets

x_test = test_data[test_data.columns[:n_lags+1]]/normalization  # Lagged features
y_test = test_data[test_data.columns[n_lags+1:]]/normalization  # Future targets

if len(x_test)<forecast_horizon:    
    print("WARNING: Forecast_horizon too long relative to test/training split")

#%% define functions
import src.pymc_functions as fu_pymc

activation = 'swish'
n_hidden_layer1 = 20
n_hidden_layer2 = 20


model = fu_pymc.construct_pymc_tlp(
    x_train = x_train,
    y_train = y_train,
    n_hidden_layer1 = n_hidden_layer1,
    n_hidden_layer2 = n_hidden_layer2,
    activation = activation
    )


draws = 500
with model:
    trace = pm.sample(
        tune = 100,
        draws = draws,
        chains = 1,
        #target_accept = 0.9,
        return_inferencedata = True
        )

#%% Test with applying coefficients manually

test_input = x_test.iloc[forecast_horizon:forecast_horizon+1]

predictions = fu_pymc.predict_tlp(
        trace = trace,
        x_test = test_input,
        activation = activation
        )

N_train = x_train.shape[1]+x_train.shape[0]-2
N_test = len(df_passengers)-N_train
cut = N_train+forecast_horizon+2

plt.figure(figsize = (10,5))
plt.plot(np.arange(N_train+1),df_passengers['Passengers'].iloc[:N_train+1], label = "Training Data")
for i in range(draws):
    plt.plot(
        np.arange(cut,cut+forecast_horizon),
        predictions[:,i]*normalization,
        color = "red",
        alpha = 0.1
        )
plt.plot(
    np.arange(cut,cut+forecast_horizon),
    predictions.mean(axis = 1)*normalization,
    label = "Model predictions",
    color = "black"
    )

plt.plot(np.arange(N_train,N_train+N_test),df_passengers['Passengers'].iloc[N_train:], label = "Test Data", color = "Green")
plt.ylabel("Passengers", fontsize=18)
plt.xlabel("Month", fontsize=18)
plt.xticks( fontsize=18)
plt.yticks( fontsize=18)
plt.minorticks_on()
plt.grid(which='both', linestyle='--', linewidth=0.5)  # Customize grid appearance
plt.legend(loc='upper left',fontsize=18)


# %% Test with pm.sample_posterior_predictive

"""
The result is very close to the manual, but not the same! There is randomness
in this result, since it is a random draw. However, the result should be
deterministic given a fixed set of samples. This is not so, casting the
workings of the backend into doubt..

If you sample the distribution infinitely, it should be static.

The problem becomes clear, if you try to plot predictions for several time
steps in high resolution. In this case you will find random noise with
magnitude determined by the number of samples. You do not have this artefact
when using the coefficients manually..

"""
test_input = x_test.iloc[forecast_horizon:forecast_horizon+1]

with model:
    pm.set_data({'x': test_input})
    posterior_predictive = pm.sample_posterior_predictive(
        trace = trace,
        predictions = True
        )

predictions2 = posterior_predictive.predictions['y_obs'].mean((('chain'))).values

plt.figure(figsize = (10,5))
plt.plot(np.arange(N_train+1),df_passengers['Passengers'].iloc[:N_train+1], label = "Training Data")
plt.plot(np.arange(N_train,N_train+N_test),df_passengers['Passengers'].iloc[N_train:], label = "Test Data", color = "Green")

plt.plot(
    np.arange(cut,cut+forecast_horizon),
    predictions.mean(axis = 1)*normalization,
    label = "Applying coefficients manually",
    color = "black"
    )
plt.plot(
    np.arange(cut,cut+forecast_horizon),
    predictions2.mean(axis = 0)[0]*normalization,
    label = "using sample_posterior_predictive",
    color = "cyan"
    )
plt.ylabel("Passengers", fontsize=18)
plt.xlabel("Month", fontsize=18)
plt.xticks( fontsize=18)
plt.yticks( fontsize=18)
plt.minorticks_on()
plt.grid(which='both', linestyle='--', linewidth=0.5)  # Customize grid appearance
plt.legend(loc='upper left',fontsize=18)


plt.figure(figsize = (10,5))
for i in range(draws):
    plt.plot(
        predictions[:,i]*normalization,
        color = "red",
        alpha = 0.1
        )
plt.plot(
    predictions.mean(axis = 1)*normalization,
    label = "Model predictions",
    color = "black"
    )

for i in range(draws):
    plt.plot(
        predictions2[i][0]*normalization,
        color = "green",
        alpha = 0.1
        )
plt.plot(
    predictions2.mean(axis = 0)[0]*normalization,
    label = "Model predictions",
    color = "cyan"
    )
plt.ylabel("Passengers", fontsize=18)
plt.xlabel("Month", fontsize=18)
plt.xticks( fontsize=18)
plt.yticks( fontsize=18)
plt.minorticks_on()
plt.grid(which='both', linestyle='--', linewidth=0.5)  # Customize grid appearance
plt.legend(loc='upper left',fontsize=18)




#%% Test if the output is consistent

"""
When making a model prediction, we are usually interested in the expected
action of Nature, which often is equal to the expectation of some posterior
distributution. If we fit a model, we can draw samples for the coefficients
for the model and compute a model output for each coefficient. The expectation
is then the arithmetic mean of the model outputs, corresponding to the mean of
the model over different coefficients, sampled from the posterior.

Given a a training session of the model, a given, fixed set of coefficients are
obtained. This corresponds then to a fixed estimate of the mean over the model
outputs. This is usually the quantity of interest. This does not seem to be 
trivially available from pymc. I can get it by extracting the coefficients! 
However, I would have wished there would be an easier way, since pymc has all
the tools available. I thought pm.sample_posterior_predictive() did this, but
it does not. There appear to be randomness in this, corresponding to sampling
the posterior. Thus, given the same input, the sampler will yield different
outputs. What do I do when I am only interested in the mean?

"""


M = 10
test_input = x_test.iloc[forecast_horizon:forecast_horizon+1]
test = np.ones((M,test_input.shape[1]))
for i in range(M):
    test[i] = test_input


with model:
    pm.set_data({'x': test_input})
    posterior_predictive3 = pm.sample_posterior_predictive(
        posterior,
        return_inferencedata = True,
        predictions = True
        )

predictions3 = posterior_predictive3.predictions['y_obs'].mean((('chain','draw'))).values

print("")
print("Is the model consistent? ", len(set(predictions3.sum(axis = 1))) == 1)

print(posterior_predictive3.predictions['y_obs'].mean(('chain','draw')).values)

