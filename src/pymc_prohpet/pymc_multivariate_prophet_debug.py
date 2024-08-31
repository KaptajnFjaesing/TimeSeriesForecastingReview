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


series = y_train.values[:,0]

plt.figure()
plt.plot(x_train,series)


# FFT of the time series
fft_result = np.fft.fft(series)
fft_magnitude = np.abs(fft_result)
fft_freqs = np.fft.fftfreq(len(series), x_train[1] - x_train[0])

# Get positive frequencies
positive_freqs = fft_freqs[fft_freqs > 0]
positive_magnitudes = fft_magnitude[fft_freqs > 0]


# Find the dominant component
max_magnitude = np.max(positive_magnitudes)
max_index = np.argmax(positive_magnitudes)
dominant_frequency = positive_freqs[max_index]
dominant_period = 1 / dominant_frequency

# Define the fraction K
K = 0.5  # Example: 50% of the maximum magnitude

# Find components that are more than K fraction of the maximum
significant_indices = np.where(positive_magnitudes >= K * max_magnitude)[0]
significant_frequencies = positive_freqs[significant_indices]
significant_periods = 1 / significant_frequencies

print(significant_periods)

#%%

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
        season_parameter = pm.Normal(f'season_parameter_{name}', mu=0, sigma=2, shape = dimension)
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
        n_fourier_components_shared: int,
        n_fourier_components_individual: int,
        shared_periods: list,
        individual_periods: list,
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
        
        
        seasonalities_shared = pm.math.stack([
            add_fourier_term(
                model=model,
                x=x,
                n_fourier_components=n_fourier_components_shared,
                name=f"shared_{i}",  # Iterating over name by appending the index
                dimension=1,
                seasonality_period_baseline=seasonality_period,
            )
            for i, seasonality_period in enumerate(shared_periods)
        ])


        seasonalities_individual = pm.math.stack([
            add_fourier_term(
                model=model,
                x=x,
                n_fourier_components=n_fourier_components_individual,
                name=f"individual_{i}",  # Iterating over name by appending the index
                dimension=n_time_series,
                seasonality_period_baseline=seasonality_period,
            )
            for i, seasonality_period in enumerate(individual_periods)
        ])

        
        prediction = trend*(1+pm.math.sum(seasonalities_shared, axis=0)) + pm.math.sum(seasonalities_individual, axis=0) # (n_obs, n_time_series)
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


n_fourier_components_shared = 10
n_fourier_components_individual = 5

n_changepoints = 2

print("n_obs: ", x_train.shape[0])
print("n_time_series: ",y_train.shape[1])
print("n_changepoints: ", n_changepoints)


model =  pymc_prophet(
        x_train = x_train,
        y_train = y_train,
        n_fourier_components_shared = n_fourier_components_shared,
        n_fourier_components_individual =  n_fourier_components_individual,
        shared_periods = [significant_periods[2]],
        individual_periods = [significant_periods[2]],
        n_changepoints = n_changepoints
        )

# Training
with model:
    #step = pm.NUTS(step_scale=0.4)  # Manually set the step size
    trace = pm.sample(
        tune = 100,
        draws = 500, 
        chains = 4,
        cores = 1,
        #target_accept=0.9 # default is 0.8
        )
    
#%%

import arviz as az

x_total = (df_passengers.index.values-x_train_mean)/x_train_std
y_total = (df_passengers[['Passengers']].values-y_train_mean)/y_train_std


X = x_total

with model:
    pm.set_data({'x':X})
    posterior_predictive = pm.sample_posterior_predictive(trace = trace, predictions=True)

preds_out_of_sample = posterior_predictive.predictions_constant_data.sortby('x')['x']
model_preds = posterior_predictive.predictions.sortby(preds_out_of_sample)

hdi_values = az.hdi(model_preds)["obs"].transpose("hdi", ...)

plt.figure()
colors = ['blue','red','green']

for i in range(y_train.shape[1]):
    plt.plot(
        preds_out_of_sample,
        (model_preds["obs"].mean(("chain", "draw")).T)[i],
        color = colors[i]
        )
    plt.fill_between(
        preds_out_of_sample.values,
        hdi_values[0].values[:,i],  # lower bound of the HDI
        hdi_values[1].values[:,i],  # upper bound of the HDI
        color=colors[i],   # color of the shaded region
        alpha=0.4,      # transparency level of the shaded region
    )

plt.plot(x_train,y_train)
plt.plot(x_test,y_test)