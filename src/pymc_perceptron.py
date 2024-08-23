#%% setup data
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

def predict(x, W_in, b_in, W_hidden, b_hidden, W_out, b_out, activation='relu'):
    """
    Predicts outputs using a two-layer perceptron with the given weights and biases.
    
    Parameters:
    x (np.ndarray): Input data of shape (n_samples, n_features).
    W_in (np.ndarray): Weights for the first hidden layer of shape (n_hidden_layer1, n_features).
    b_in (np.ndarray): Biases for the first hidden layer of shape (n_hidden_layer1,).
    W_hidden (np.ndarray): Weights for the second hidden layer of shape (n_hidden_layer2, n_hidden_layer1).
    b_hidden (np.ndarray): Biases for the second hidden layer of shape (n_hidden_layer2,).
    W_out (np.ndarray): Weights for the output layer of shape (1, n_hidden_layer2).
    b_out (np.ndarray): Biases for the output layer of shape (1,).
    activation (str): The activation function to use ('relu' or 'swish').
    
    Returns:
    np.ndarray: Predicted outputs.
    """
    
    # Compute the first hidden layer
    linear_layer1 = np.dot(x, W_in.T) + b_in.T
    if activation == 'relu':
        hidden_layer1 = np.maximum(linear_layer1, 0)
    elif activation == 'swish':
        hidden_layer1 = linear_layer1 * (1 / (1 + np.exp(-linear_layer1)))
    else:
        raise ValueError("Unsupported activation function")
    
    # Compute the second hidden layer
    linear_layer2 = np.dot(hidden_layer1, W_hidden.T) + b_hidden.T
    if activation == 'relu':
        hidden_layer2 = np.maximum(linear_layer2, 0)
    elif activation == 'swish':
        hidden_layer2 = linear_layer2 * (1 / (1 + np.exp(-linear_layer2)))
    else:
        raise ValueError("Unsupported activation function")
    
    # Compute the output layer
    output = np.dot(hidden_layer2, W_out.T) + b_out.T
    
    return output

def construct_pymc_model(
        x_train: np.array,
        y_train: np.array,
        n_hidden_layer1: int = 10,
        n_hidden_layer2: int = 10,
        activation: str ='relu'
        ):

    with pm.Model() as model:
        n_features = x_train.shape[1]
        forecast_horizon = y_train.shape[1]
        print("Forecast horizon ", forecast_horizon)
        x = pm.Data("x", x_train)
        print(f"input shape: {x.shape.eval()}")
        # Priors for the weights and biases of the hidden layer 1
        precision_hidden_w1 = pm.Gamma('precision_hidden_w1', alpha=2.4, beta=3)
        precision_hidden_b1 = pm.Gamma('precision_hidden_b1', alpha=2.4, beta=3)
        
        # Hidden layer 1 weights and biases
        W_hidden1 = pm.Normal(
            'W_hidden1',
            mu=0,
            sigma=1/np.sqrt(precision_hidden_w1),
            shape=(n_hidden_layer1, n_features)
            )
        b_hidden1 = pm.Normal(
            'b_hidden1',
            mu=0,
            sigma=1/np.sqrt(precision_hidden_b1),
            shape=(n_hidden_layer1,)
            )
        
        # Compute the hidden layer 1 outputs
        linear_layer1 = pm.math.dot(x, W_hidden1.T) + b_hidden1
        
        # Print shapes of intermediate computations
        print(f"W_hidden1 shape: {W_hidden1.shape.eval()}")
        print(f"b_hidden1 shape: {b_hidden1.shape.eval()}")
        print(f"linear_layer1 shape: {linear_layer1.shape.eval()}")

        if activation == 'relu':
            hidden_layer1 = pm.math.maximum(linear_layer1, 0)
        elif activation == 'swish':
            hidden_layer1 = linear_layer1 * pm.math.sigmoid(linear_layer1)
        else:
            raise ValueError("Unsupported activation function")

        # Priors for the weights and biases of the hidden layer 2
        precision_hidden_w2 = pm.Gamma('precision_hidden_w2', alpha=2.4, beta=3)
        precision_hidden_b2 = pm.Gamma('precision_hidden_b2', alpha=2.4, beta=3)
        
        # Hidden layer 2 weights and biases
        W_hidden2 = pm.Normal(
            'W_hidden2',
            mu=0,
            sigma=1/np.sqrt(precision_hidden_w2),
            shape=(n_hidden_layer2, n_hidden_layer1)
            )
        b_hidden2 = pm.Normal(
            'b_hidden2',
            mu=0,
            sigma=1/np.sqrt(precision_hidden_b2),
            shape=(n_hidden_layer2,)
            )
        
        # Compute the hidden layer 2 outputs
        linear_layer2 = pm.math.dot(hidden_layer1, W_hidden2.T) + b_hidden2

        print(f"W_hidden2 shape: {W_hidden2.shape.eval()}")
        print(f"b_hidden2 shape: {b_hidden2.shape.eval()}")
        print(f"linear_layer2 shape: {linear_layer2.shape.eval()}")

        if activation == 'relu':
            hidden_layer2 = pm.math.maximum(linear_layer2, 0)
        elif activation == 'swish':
            hidden_layer2 = linear_layer1 * pm.math.sigmoid(linear_layer2)
        else:
            raise ValueError("Unsupported activation function")
        
        # Priors for the weights and biases of the output layer
        precision_output_w = pm.Gamma('precision_output_w', alpha=2.4, beta=3)
        precision_output_b = pm.Gamma('precision_output_b', alpha=2.4, beta=3)
        
        # Output layer weights and biases
        W_output = pm.Normal('W_output', mu=0, sigma=1/np.sqrt(precision_output_w), shape=(forecast_horizon, n_hidden_layer2))
        b_output = pm.Normal('b_output', mu=0, sigma=1/np.sqrt(precision_output_b), shape=(forecast_horizon,))
        
        print(f"hidden_layer2 shape: {hidden_layer2.shape.eval()}")

        # Compute the output (regression prediction)
        y_pred = pm.math.dot(hidden_layer2, W_output.T) + b_output
        
        # Print shape of final prediction
        print(f"W_output shape: {W_output.shape.eval()}")
        print(f"b_output shape: {b_output.shape.eval()}")
        print(f"y_pred shape: {y_pred.shape.eval()}")

        # Likelihood (using Normal distribution for regression)
        precision_obs = pm.Gamma('precision_obs', alpha=2.4, beta=3)
        pm.Normal(
            'y_obs',
            mu = y_pred,
            sigma = 1/np.sqrt(precision_obs),
            observed = y_train
            )

        return model

model = construct_pymc_model(x_train = x_train, y_train = y_train)


# %% train model
with model:
    posterior = pm.sample(tune=100, draws=1000, chains=1)

# %% plot results
W_in = posterior.posterior['W_hidden1'].mean((('chain'))).values
b_in = posterior.posterior['b_hidden1'].mean((('chain'))).values

W_hidden = posterior.posterior['W_hidden2'].mean((('chain'))).values
b_hidden = posterior.posterior['b_hidden2'].mean((('chain'))).values

W_out = posterior.posterior['W_output'].mean((('chain'))).values
b_out = posterior.posterior['b_output'].mean((('chain'))).values


preds = np.zeros((forecast_horizon,len(W_in)))
for i in range(len(W_in)):
    preds[:,i] = predict(
        x = x_test.iloc[forecast_horizon:forecast_horizon+1],
        W_in = W_in[i],
        b_in = b_in[i],
        W_hidden = W_hidden[i],
        b_hidden = b_hidden[i],
        W_out = W_out[i],
        b_out = b_out[i],
        activation='relu'
        )


"""
The logic here is that I want to test the model on data it has never seen
during training. For this reason, the test data point is one point beyond the 
last training target value; 

x_test.iloc[forecast_horizon:forecast_horizon+1]

This is element 104. The target predicts from one timestep ahead and 26
further. This means from [105,130]. 'cut' is 105.


"""


N_train = x_train.shape[1]+x_train.shape[0]-2
N_test = len(df_passengers)-N_train

cut = N_train+forecast_horizon+2

plt.figure(figsize = (20,10))
plt.plot(np.arange(N_train+1),df_passengers['Passengers'].iloc[:N_train+1], label = "Training Data")
for i in range(len(W_in)):
    plt.plot(
        np.arange(cut,cut+forecast_horizon),
        preds[:,i]*normalization,
        color = "red",
        alpha = 0.1
        )
plt.plot(
    np.arange(cut,cut+forecast_horizon),
    preds.mean(axis = 1)*normalization,
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

# %% 

with model:
    pm.set_data({'x': x_test.iloc[forecast_horizon:forecast_horizon+1]})
    posterior_predictive = pm.sample_posterior_predictive(posterior, extend_inferencedata=True, predictions=True)

# %%
predictions = posterior_predictive.predictions['y_obs'].mean((('chain'))).values*normalization


preds.mean(axis = 1)*normalization