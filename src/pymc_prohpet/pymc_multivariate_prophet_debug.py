from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pytensor as pt

df_passengers = load_passengers_data()  # Load the data

training_split = int(len(df_passengers)*0.7)
 
x_train_unnormalized = df_passengers.index[:training_split].values
y_train_unnormalized = df_passengers[['Passengers']].iloc[:training_split]

x_train_mean = x_train_unnormalized.mean()
x_train_std = x_train_unnormalized.std()

y_train_mean = y_train_unnormalized.mean().values
y_train_std = y_train_unnormalized.std().values

x_train = (x_train_unnormalized-x_train_mean)/x_train_std
y_train = (y_train_unnormalized-y_train_mean)/y_train_std

x_test = (df_passengers.index[training_split:].values-x_train_mean)/x_train_std
y_test = (df_passengers[['Passengers']].iloc[training_split:].values-y_train_mean)/y_train_std



def create_fourier_features(
        x: np.ndarray,
        n_fourier_components: int,
        seasonality_period: np.ndarray
        ):

    frequency_component = pt.tensor.as_tensor_variable(2 * np.pi * (np.arange(n_fourier_components)+1) * x[:, None]) # (n_obs, n_fourier_components)
    t = frequency_component[:, :, None] / seasonality_period[None, None, :] # (n_obs, n_fourier_components, n_time_series)
    fourier_features = pm.math.concatenate((pt.tensor.cos(t), pt.tensor.sin(t)), axis=1)  # (n_obs, 2*n_fourier_components, n_time_series)
    return fourier_features


def add_fourier_term(
        model,
        x: np.array,
        n_fourier_components: int,
        name: str,
        dimension: int,
        seasonality_period_baseline: float,
        ):
    with model:
        fourier_coefficients = pm.Normal(
            f'fourier_coefficients_{name}',
            mu=0,
            sigma=5,
            shape=2 * n_fourier_components
        )
        season_parameter = pm.Normal(f'season_parameter_{name}', mu=0, sigma=1, shape = dimension)
        seasonality_period = seasonality_period_baseline * pm.math.exp(season_parameter)
        
        fourier_features = create_fourier_features(
            x=x,
            n_fourier_components=n_fourier_components,
            seasonality_period=seasonality_period
        )
    
    output = pt.tensor.sum(fourier_features * fourier_coefficients[None,:,None], axis=1) # (n_obs, n_time_series)
    return output


def pymc_prophet(
        x_train: np.array,
        y_train: np.array,
        seasonality_period_baseline_shared: float,
        n_fourier_components_shared: int,
        seasonality_period_baseline_individual: float,
        n_fourier_components_individual: int,
        n_changepoints: int = 10
        ):
    
    # Define the model
    with pm.Model() as model:
        # Data variables
        n_time_series = y_train.shape[1]
        x = pm.Data('x', x_train, dims= ['n_obs'])
        y = pm.Data('y', y_train, dims= ['n_obs_target', 'n_time_series'])
    
        s = pt.tensor.linspace(0, pm.math.max(x), n_changepoints+2)[1:-1]
        A = (x[:, None] > s)*1.
    
        # Growth model parameters
        k = pm.Normal('k', mu=0, sigma=5, shape=n_time_series)  # Shape matches time series
        m = pm.Normal('m', mu=0, sigma=5, shape=n_time_series)
        delta = pm.Laplace('delta', mu=0, b=0.5, shape = (n_time_series, n_changepoints))
        gamma = -s * delta  # Shape is (n_time_series, n_changepoints)
        # Trend calculation
        growth = k + pm.math.dot(A, delta.T)  # Shape: (n_obs, n_time_series)
        offset = m + pm.math.dot(A, gamma.T)  # Shape: (n_obs, n_time_series)
    
        trend = growth*x[:, None] + offset  # Shape: (n_obs, n_time_series)
        
        # shared Seasonality model
        seasonality_shared = add_fourier_term(
                model = model,
                x = x,
                n_fourier_components = n_fourier_components_shared,
                name = "shared",
                dimension = 1,
                seasonality_period_baseline = seasonality_period_baseline_shared,
                ) # (n_obs,1)
        
        
        seasonality_individual = add_fourier_term(
                model = model,
                x = x,
                n_fourier_components = n_fourier_components_individual,
                name = "individual",
                dimension = n_time_series,
                seasonality_period_baseline = seasonality_period_baseline_individual,
                ) # (n_obs, n_time_series)
    
        prediction = trend*(1+seasonality_individual) + seasonality_shared # (n_obs, n_time_series)
        error = pm.HalfCauchy('sigma', beta=1, dims = 'n_time_series')
        pm.Normal(
            'obs',
            mu = prediction,
            sigma = error,
            observed = y,
            dims = ['n_obs', 'n_time_series']
        )
    return model

#%%

"""
FIX THE CORRECT PERIOD !


"""

n_fourier_components_shared = 10
seasonality_period_baseline_shared = 10

n_fourier_components_individual = 10
seasonality_period_baseline_individual = 10

n_changepoints = 11

print("n_obs: ", x_train.shape[0])
print("n_time_series: ",y_train.shape[1])
print("n_changepoints: ", n_changepoints)


model =  pymc_prophet(
        x_train = x_train,
        y_train = y_train,
        seasonality_period_baseline_shared = seasonality_period_baseline_shared,
        n_fourier_components_shared = n_fourier_components_shared,
        seasonality_period_baseline_individual = seasonality_period_baseline_individual,
        n_fourier_components_individual =  n_fourier_components_individual,
        n_changepoints = n_changepoints
        )

# Training
with model:
    trace = pm.sample(
        tune = 100,
        draws = 500, 
        chains = 1,
        cores = 1
        )
    
#%%


import arviz as az

X = x_train

with model:
    pm.set_data({'x':X})
    posterior_predictive = pm.sample_posterior_predictive(trace = trace, predictions=True)

preds_out_of_sample = posterior_predictive.predictions_constant_data.sortby('x')['x']
model_preds = posterior_predictive.predictions.sortby(preds_out_of_sample)

plt.figure()
plt.plot(
    preds_out_of_sample,
    model_preds["obs"].mean(("chain", "draw"))
    )
hdi_values = az.hdi(model_preds)["obs"].transpose("hdi", ...)

plt.fill_between(
    preds_out_of_sample.values,
    hdi_values[0].values,  # lower bound of the HDI
    hdi_values[1].values,  # upper bound of the HDI
    color="gray",   # color of the shaded region
    alpha=0.4,      # transparency level of the shaded region
)
plt.plot(x_train,y_train)
plt.plot(x_test,y_test, color = "blue")