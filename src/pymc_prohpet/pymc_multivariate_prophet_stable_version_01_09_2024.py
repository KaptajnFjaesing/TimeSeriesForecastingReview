from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pytensor as pt

df_passengers = load_passengers_data()  # Load the data

# Generate mock time series
np.random.seed(42)  # For reproducibility
trend = -80 + 1.4 * np.arange(len(df_passengers))
noise = 1*np.random.normal(0, 10, len(df_passengers))
df_passengers['Mock_Passengers'] = (df_passengers['Passengers']*(0.6-0.01*np.arange(len(df_passengers))) + trend + noise).astype(int)


training_split = int(len(df_passengers)*0.7)
 
x_train_unnormalized = df_passengers.index[:training_split].values
y_train_unnormalized = df_passengers[['Passengers','Mock_Passengers']].iloc[:training_split]

x_train_mean = x_train_unnormalized.mean()
x_train_std = x_train_unnormalized.std()

y_train_mean = y_train_unnormalized.mean().values
y_train_std = y_train_unnormalized.std().values

x_train = (x_train_unnormalized-x_train_mean)/x_train_std
y_train = (y_train_unnormalized-y_train_mean)/y_train_std

x_test = (df_passengers.index[training_split:].values-x_train_mean)/x_train_std
y_test = (df_passengers[['Passengers','Mock_Passengers']].iloc[training_split:]-y_train_mean)/y_train_std

x_total = (df_passengers.index.values-x_train_mean)/x_train_std
y_total = (df_passengers[['Passengers','Mock_Passengers']]-y_train_mean)/y_train_std


plt.figure()
plt.plot(df_passengers['Date'], df_passengers[['Passengers','Mock_Passengers']])

series = y_train.values[:,0]

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



# %%

def create_fourier_features(
        x: np.ndarray,
        n_fourier_components: int,
        seasonality_period: np.ndarray
        ):

    frequency_component = pt.tensor.as_tensor_variable(2 * np.pi * (np.arange(n_fourier_components)+1) * x[:, None]) # (n_obs, n_fourier_components)
    t = frequency_component[:, :, None] / seasonality_period # (n_obs, n_fourier_components, n_time_series)
    fourier_features = pm.math.concatenate((pt.tensor.cos(t), pt.tensor.sin(t)), axis=1)  # (n_obs, 2*n_fourier_components, n_time_series)
    return fourier_features

def add_linear_term(
        model,
        x,
        name,
        k_est: float,
        n_time_series: int,
        n_changepoints: int,
        prior_alpha: float,
        prior_beta: float
        ):
    with model:
        s = pt.tensor.linspace(0, pm.math.max(x), n_changepoints+2)[1:-1]
        A = (x[:, None] > s)*1.
    
        # Growth model parameters
        precision_k = pm.Gamma(
            f'precision_{name}_k',
            alpha = prior_alpha,
            beta = prior_beta
            )
        k = pm.Normal(
            f'{name}_k',
            mu = k_est,
            sigma = 1/np.sqrt(precision_k),
            shape = n_time_series
            )  # Shape matches time series
        delta = pm.Normal(f'{name}_delta', mu=0, sigma = 1, shape = (n_time_series, n_changepoints)) # The parameter delta represents the magnitude and direction of the change in growth rate at each changepoint in a piecewise linear model.

        # Offset model parameters
        precision_m = pm.Gamma(
            f'precision_{name}_m',
            alpha = prior_alpha,
            beta = prior_beta
            )
        m = pm.Normal(f'{name}_m', mu=0, sigma = 1/np.sqrt(precision_m), shape=n_time_series)
        
    return (k + pm.math.dot(A, delta.T))*x[:, None] + m + pm.math.dot(A, (-s * delta).T)


def add_fourier_term(
        model,
        x: np.array,
        n_fourier_components: int,
        name: str,
        dimension: int,
        seasonality_period_baseline: float,
        prior_alpha: float,
        prior_beta: float
        ):
    with model:
        fourier_coefficients = pm.Normal(
            f'fourier_coefficients_{name}',
            mu=0,
            sigma=5,
            shape= (2 * n_fourier_components, dimension)
        )
        
        seasonality_period = pm.Normal(f'season_parameter_{name}', mu = seasonality_period_baseline, sigma = 0.01)
        
        fourier_features = create_fourier_features(
            x=x,
            n_fourier_components=n_fourier_components,
            seasonality_period=seasonality_period
        )
            
    return pm.math.sum(fourier_features * fourier_coefficients[None,:,:], axis=1)


def pymc_prophet(
        x_train: np.array,
        y_train: np.array,
        seasonality_period_baseline: float,
        n_changepoints: int = 10,
        n_fourier_components: int = 10
        ):
    
    prior_alpha = 1
    prior_beta = 1
    
    with pm.Model() as model:
        n_time_series = y_train.shape[1]
        k_est = (y_train.values[-1][0]-y_train.values[0][0])/(x_train[-1]-x_train[0])
        x = pm.Data('x', x_train, dims= ['n_obs'])
        y = pm.Data('y', y_train, dims= ['n_obs_target', 'n_time_series'])
        
        
        """
        s = pt.tensor.linspace(0, pm.math.max(x), n_changepoints+2)[1:-1]
        A = (x[:, None] > s)*1.
        
        k = pm.Normal('k', mu = 0, sigma = 5)
        m = pm.Normal('m', mu = 0, sigma = 5)
        delta = pm.Laplace('delta', mu = 0, b = 0.5, shape = n_changepoints) # the b parameter has a big impact on the ability to over/underfit
        gamma = -s*delta
        growth = k+pm.math.dot(A,delta)
        offset = m+pm.math.dot(A,gamma)
        trend = growth*x+offset
        """
        
        linear_term = add_linear_term(
                model = model,
                x = x,
                name = 'hest',
                k_est = k_est,
                n_time_series = n_time_series,
                n_changepoints = n_changepoints,
                prior_alpha = prior_alpha,
                prior_beta = prior_beta
                )
        
        
        seasonality_1 = add_fourier_term(
                model = model,
                x = x,
                n_fourier_components = n_fourier_components,
                name = 'ko',
                dimension = 1,
                seasonality_period_baseline = seasonality_period_baseline,
                prior_alpha = prior_alpha,
                prior_beta =prior_beta
                )
        
        seasonality_2 = add_fourier_term(
                model = model,
                x = x,
                n_fourier_components = n_fourier_components,
                name = 'ko2',
                dimension = 1,
                seasonality_period_baseline = seasonality_period_baseline,
                prior_alpha = prior_alpha,
                prior_beta =prior_beta
                )
        
        """
        fourier_coefficients_1 = pm.Normal('fourier_coefficients1', mu = 0, sigma = 5, shape = 2*n_fourier_components) # parameters for each sine and cosine
        fourier_coefficients_2 = pm.Normal('fourier_coefficients2', mu = 0, sigma = 5, shape = 2*n_fourier_components) # parameters for each sine and cosine
        season_parameter = pm.Normal('season_parameter', mu = 0, sigma = 1)
        seasonality_period = seasonality_period_baseline*pm.math.exp(season_parameter)
        
        fourier_sine_terms = create_fourier_features(
            t = x,
            n_fourier_components = n_fourier_components,
            seasonality_period = seasonality_period
            )
        seasonality_1 = pm.math.dot(fourier_sine_terms,fourier_coefficients_1)
        seasonality_2 = pm.math.dot(fourier_sine_terms,fourier_coefficients_2)
        """
        
        print(seasonality_2.shape.eval(),linear_term.shape.eval())
        
        prediction = linear_term[:,0]*(1+seasonality_2[:,0]) + seasonality_1[:,0]
        
        
        precision_obs = pm.Gamma(
            'precision_obs',
            alpha = prior_alpha,
            beta = prior_beta,
            dims = 'n_time_series'
            )

        pm.Normal(
            'obs',
            mu = prediction[:,None],
            sigma = 1/np.sqrt(precision_obs),
            observed = y,
            dims = ['n_obs', 'n_time_series']
        )
    return model

#%%


n_fourier_components_shared = 5
seasonality_period_baseline_shared = max(significant_periods)

n_fourier_components_individual = 5
seasonality_period_baseline_individual = max(significant_periods)

n_changepoints = 10

print("n_obs: ", x_train.shape[0])
print("n_time_series: ",y_train.shape[1])
print("n_changepoints: ", n_changepoints)


target = y_train[['Passengers']]


model =  pymc_prophet(
        x_train = x_train,
        y_train = target,
        seasonality_period_baseline = min(significant_periods),
        n_fourier_components = 10,
        n_changepoints = n_changepoints
        )


#%%
# Training


with model:
    steps = [pm.NUTS(), pm.HamiltonianMC(), pm.Metropolis()]
    trace = pm.sample(
        tune = 500,
        draws = 2000, 
        chains = 4,
        cores = 1,
        step = steps[0]
        #target_accept = 0.9
        )

#%%



pm.plot_trace(
    trace,
    )



#%%

import arviz as az


X = x_total

with model:
    pm.set_data({'x':X})
    posterior_predictive = pm.sample_posterior_predictive(trace = trace, predictions=True)

preds_out_of_sample = posterior_predictive.predictions_constant_data.sortby('x')['x']
model_preds = posterior_predictive.predictions.sortby(preds_out_of_sample)

hdi_values = az.hdi(model_preds)["obs"].transpose("hdi", ...)



for i in range(target.shape[1]):
    plt.figure()
    plt.title(target.columns[i])
    plt.plot(
        preds_out_of_sample,
        (model_preds["obs"].mean(("chain", "draw")).T)[i],
        color = 'blue'
        )
    plt.fill_between(
        preds_out_of_sample.values,
        hdi_values[0].values[:,i],  # lower bound of the HDI
        hdi_values[1].values[:,i],  # upper bound of the HDI
        color= 'blue',   # color of the shaded region
        alpha=0.4,      # transparency level of the shaded region
    )

    plt.plot(x_train,target[target.columns[i]], color = 'red')
    plt.plot(x_test,y_test[y_test.columns[i]], color = 'green')





