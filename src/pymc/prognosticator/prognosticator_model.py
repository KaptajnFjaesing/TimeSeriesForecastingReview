"""
Created on Mon Sep 16 07:36:48 2024

@author: Jonas Petersen
"""

import numpy as np
from pathlib import Path
import json
import pymc as pm
import pandas as pd
import arviz as az
from typing import (
    Dict,
    Tuple,
    Union,
    Any
    )

# Import from other modules
from src.config import (
    get_default_model_config,
    get_default_sampler_config,
    serialize_model_config,
)
from src.model_components import (
    swish
)
from src.utils import (
    generate_hash_id,
    normalize_training_data
    )


class Prognosticator:

    def __init__(
        self,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        model_name: str = "SorcererModel",
        version: str = None
    ):
        # Initialize configurations
        sampler_config = (
            get_default_sampler_config() if sampler_config is None else sampler_config
        )
        self.sampler_config = sampler_config
        model_config = get_default_model_config() if model_config is None else model_config
        self.model_config = model_config  # parameters for priors, etc.
        self.model = None  # Set by build_model
        self.idata: az.InferenceData | None = None  # idata is generated during fitting
        self.posterior_predictive: az.InferenceData
        self.model_name = model_name
        self.version = version
        self.method = "NUTS"
        self.map_estimate = None
        self.x_training_min = None
        self.x_training_max = None
        self.y_training_min = None
        self.y_training_max = None
        
    def build_model(
            self,
            X,
            y,
            n_hidden_layer1,
            n_hidden_layer2,
            activation,
            **kwargs
            ):
        """
        Builds the PyMC model based on the input data and configuration.
        """
   
        # Define PyMC model
        with pm.Model() as self.model:
            n_features = X.shape[1]
            forecast_horizon = y.shape[2]
            
            x = pm.Data("x", X, dims = ['number_of_input_observations', 'feature_dim'])
            y = pm.Data("y", y, dims = ['number_of_time_series','number_of_output_observations', 'forecast_horizon'])
            # Priors for the weights and biases of the hidden layer 1
            precision_hidden_w1 = pm.Gamma('precision_hidden_w1', alpha = self.model_config["w_input_alpha_prior"], beta = self.model_config["w_input_beta_prior"])
            precision_hidden_b1 = pm.Gamma('precision_hidden_b1', alpha = self.model_config["b_input_alpha_prior"], beta = self.model_config["b_input_beta_prior"])
            
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
                hidden_layer1 = swish(linear_layer1)
                #hidden_layer1 = linear_layer1 * pm.math.sigmoid(linear_layer1)
            else:
                raise ValueError("Unsupported activation function")

            # Priors for the weights and biases of the hidden layer 2
            precision_hidden_w2 = pm.Gamma('precision_hidden_w2', alpha = self.model_config["w2_alpha_prior"], beta = self.model_config["w2_beta_prior"])
            precision_hidden_b2 = pm.Gamma('precision_hidden_b2', alpha = self.model_config["b2_alpha_prior"], beta = self.model_config["b2_beta_prior"])
            
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
                hidden_layer2 = swish(linear_layer2)
                #hidden_layer2 = linear_layer1 * pm.math.sigmoid(linear_layer2)
            else:
                raise ValueError("Unsupported activation function")
            
            # Priors for the weights and biases of the output layer
            precision_output_w = pm.Gamma('precision_output_w', alpha = self.model_config["w_output_alpha_prior"], beta = self.model_config["w_output_beta_prior"])
            precision_output_b = pm.Gamma('precision_output_b', alpha = self.model_config["b_output_alpha_prior"], beta = self.model_config["b_output_beta_prior"])
            
            # Output layer weights and biases
            W_output = pm.Normal('W_output', mu=0, sigma=1/np.sqrt(precision_output_w), shape=(forecast_horizon, n_hidden_layer2))
            b_output = pm.Normal('b_output', mu=0, sigma=1/np.sqrt(precision_output_b), shape=(forecast_horizon,))

            # Compute the output (regression prediction)
            target_mean = pm.math.dot(hidden_layer2, W_output.T) + b_output
            
            # Likelihood (using Normal distribution for regression)
            precision_obs = pm.Gamma('precision_obs', alpha = self.model_config["precision_alpha_prior"], beta = self.model_config["precision_beta_prior"])
            pm.Normal(
                'y_obs',
                mu = target_mean,
                sigma = 1/np.sqrt(precision_obs),
                observed = y,
                dims = ['number_of_time_series','number_of_input_observations', 'forecast_horizon']
                )

    def fit(
        self,
        training_data: pd.DataFrame,
        seasonality_periods: np.array,
        progressbar: bool = True,
        random_seed: pm.util.RandomState = None,
        method: str = "NUTS",
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Fits the model to the data using the specified sampler configuration.
        """
        self.method = method
        (
            X,
            self.x_training_min,
            self.x_training_max,
            y,
            self.y_training_min,
            self.y_training_max
            )  = normalize_training_data(training_data = training_data)
        self.build_model(
            X = X,
            y = y,
            seasonality_periods = seasonality_periods*(X[1]-X[0])
            )
        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)
        with self.model:
            if self.method == "MAP":
                self.map_estimate = [pm.find_MAP()]
            else:
                if self.method == "NUTS":
                    step=pm.NUTS()
                if self.method == "HMC":
                    step=pm.HamiltonianMC()
                if self.method == "metropolis":
                    step=pm.Metropolis()
                idata_temp = pm.sample(step = step, **sampler_config)
                self.idata = self.set_idata_attrs(idata_temp)

    def set_idata_attrs(self, idata=None):
        """
        Sets attributes to the inference data object for identification and metadata.
        """
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        
        idata.attrs["id"] = self.id
        idata.attrs["model_name"] = self.model_name
        idata.attrs["version"] = self.version
        idata.attrs["sampler_config"] = serialize_model_config(self.sampler_config)
        idata.attrs["model_config"] = serialize_model_config(self._serializable_model_config)

        return idata

    def sample_posterior_predictive(
        self,
        test_data,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples from the posterior predictive distribution using the fitted model.
        """
        with self.model:
            x_test = (test_data['date'].astype('int64')//10**9 - self.x_training_min)/(self.x_training_max - self.x_training_min)
            pm.set_data({'input': x_test})
            if self.method == "MAP":
                self.posterior_predictive = pm.sample_posterior_predictive(self.map_estimate, predictions=True, **kwargs)
            else:
                self.posterior_predictive = pm.sample_posterior_predictive(self.idata, predictions=True, **kwargs)
        
        preds_out_of_sample = self.posterior_predictive.predictions_constant_data.sortby('input')['input']
        model_preds = self.posterior_predictive.predictions.sortby(preds_out_of_sample)

        return preds_out_of_sample, model_preds
    
    def normalize_data(self,
                       training_data,
                       test_data
                       )-> tuple:
        if self.x_training_min is not None:
            time_series_columns = [x for x in training_data.columns if 'date' not in x]
            X_train = (training_data['date'].astype('int64')//10**9 - self.x_training_min)/(self.x_training_max - self.x_training_min)
            y_train = (training_data[time_series_columns]-self.y_training_min)/(self.y_training_max-self.y_training_min)
            
            X_test = (test_data['date'].astype('int64')//10**9 - self.x_training_min)/(self.x_training_max - self.x_training_min)
            y_test = (test_data[time_series_columns]-self.y_training_min)/(self.y_training_max-self.y_training_min)
            return (
                X_train,
                y_train,
                X_test,
                y_test
                )
        else:
            raise RuntimeError("Data can only be normalized after .fit() has been called.")
            return None
    
    def get_posterior_predictive(self) -> az.InferenceData:
        """
        Returns the posterior predictive distribution.
        """
        return self.posterior_predictive

    @property
    def id(self) -> str:
        """
        Returns a unique ID for the model instance based on its configuration.
        """
        return generate_hash_id(self.model_config, self.version, self.model_name)

    @property
    def output_var(self):
        return "target_distribution"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        return self.model_config
    
    def get_model(self):
        return self.model
    
    def get_idata(self):
        return self.idata
    
    def save(self, fname: str) -> None:
        """
        Save the model's inference data to a file.

        Parameters
        ----------
        fname : str
            The name and path of the file to save the inference data with model parameters.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model hasn't been fit yet (no inference data available).

        Examples
        --------
        This method is meant to be overridden and implemented by subclasses.
        It should not be called directly on the base abstract class or its instances.

        >>> class MyModel(ModelBuilder):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>> model = MyModel()
        >>> model.fit(data)
        >>> model.save('model_results.nc')  # This will call the overridden method in MyModel
        """
        
     
        if self.idata is not None and "posterior" in self.idata:
            if self.method == "MAP":
                raise RuntimeError("The MAP method cannot be saved.")
            file = Path(str(fname))
            self.idata.to_netcdf(str(file))
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
                

    def load(self, fname: str):
        filepath = Path(str(fname))
        self.idata = az.from_netcdf(filepath)
        # needs to be converted, because json.loads was changing tuple to list
        self.model_config = json.loads(self.idata.attrs["model_config"])
        self.sampler_config = json.loads(self.idata.attrs["sampler_config"])
        
        self.build_model(
            X = self.idata.constant_data.input,
            y = self.idata.constant_data.target
            )

