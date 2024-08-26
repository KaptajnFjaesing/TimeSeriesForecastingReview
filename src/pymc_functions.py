import numpy as np
import pymc as pm
from pymc.backends.base import MultiTrace


def construct_pymc_tlp(
        x_train: np.array,
        y_train: np.array,
        n_hidden_layer1: int = 10,
        n_hidden_layer2: int = 10,
        activation: str ='relu'
        ) -> pm.Model:

    with pm.Model() as model:
        n_features = x_train.shape[1]
        forecast_horizon = y_train.shape[1]
        
        x = pm.Data("x", x_train, dims = ['obs_dim', 'feature_dim'])
        y = pm.Data("y", y_train, dims = ['obs_dim', 'forecast_horizon'])
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

        # Compute the output (regression prediction)
        y_pred = pm.math.dot(hidden_layer2, W_output.T) + b_output
        
        # Likelihood (using Normal distribution for regression)
        precision_obs = pm.Gamma('precision_obs', alpha=2.4, beta=3)
        pm.Normal(
            'y_obs',
            mu = y_pred,
            sigma = 1/np.sqrt(precision_obs),
            observed = y,
            dims = ['obs_dim', 'forecast_horizon']
            )

        return model

def evaluate_tlp(
        x: np.array,
        W_in: np.array,
        b_in: np.array,
        W_hidden: np.array,
        b_hidden: np.array,
        W_out: np.array,
        b_out: np.array,
        activation: str ='relu'
        ) -> np.array:

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

def predict_tlp(
        posterior: MultiTrace,
        x_test: np.array,
        activation: str = 'relu'
        ) -> np.array:
    
    
    W_in = posterior.posterior['W_hidden1'].mean((('chain'))).values
    b_in = posterior.posterior['b_hidden1'].mean((('chain'))).values
    
    W_hidden = posterior.posterior['W_hidden2'].mean((('chain'))).values
    b_hidden = posterior.posterior['b_hidden2'].mean((('chain'))).values
    
    W_out = posterior.posterior['W_output'].mean((('chain'))).values
    b_out = posterior.posterior['b_output'].mean((('chain'))).values
    
    forecast_horizon = W_out.shape[1]
    
    predictions = np.zeros((forecast_horizon,len(W_in)))
    for i in range(len(W_in)):
        predictions[:,i] = evaluate_tlp(
            x = x_test,
            W_in = W_in[i],
            b_in = b_in[i],
            W_hidden = W_hidden[i],
            b_hidden = b_hidden[i],
            W_out = W_out[i],
            b_out = b_out[i],
            activation = activation
            )
    return predictions