from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pytensor as pt

df_passengers = load_passengers_data()  # Load the data

# Generate mock time series
np.random.seed(42)  # For reproducibility
trend = -80 - 0.7 * np.arange(len(df_passengers))
noise = 1*np.random.normal(0, 10, len(df_passengers))
df_passengers['Mock_Passengers'] = (-df_passengers['Passengers']*(0.6-0.004*np.sqrt(np.arange(len(df_passengers)))) + trend + noise).astype(int)


training_split = int(len(df_passengers)*0.7)
 
x_train_unnormalized = df_passengers.index[:training_split].values
y_train_unnormalized = df_passengers[['Passengers','Mock_Passengers']].iloc[:training_split]



"""

x_train_mean = x_train_unnormalized.mean()
x_train_std = x_train_unnormalized.std()*2

y_train_mean = y_train_unnormalized.mean().values
y_train_std = y_train_unnormalized.std().values*2

x_train = (x_train_unnormalized-x_train_mean)/x_train_std
y_train = (y_train_unnormalized-y_train_mean)/y_train_std

x_test = (df_passengers.index[training_split:].values-x_train_mean)/x_train_std
y_test = (df_passengers[['Passengers','Mock_Passengers']].iloc[training_split:]-y_train_mean)/y_train_std

x_total = (df_passengers.index.values-x_train_mean)/x_train_std
y_total = (df_passengers[['Passengers','Mock_Passengers']]-y_train_mean)/y_train_std

"""

x_train = (x_train_unnormalized-x_train_unnormalized.min())/(x_train_unnormalized.max()-x_train_unnormalized.min())
y_train = (y_train_unnormalized-y_train_unnormalized.min())/(y_train_unnormalized.max()-y_train_unnormalized.min())

x_test = (df_passengers.index[training_split:].values-x_train_unnormalized.min())/(x_train_unnormalized.max()-x_train_unnormalized.min())
y_test = (df_passengers[['Passengers','Mock_Passengers']].iloc[training_split:]-y_train_unnormalized.min())/(y_train_unnormalized.max()-y_train_unnormalized.min())

x_total = (df_passengers.index.values-x_train_unnormalized.min())/(x_train_unnormalized.max()-x_train_unnormalized.min())
y_total = (df_passengers[['Passengers','Mock_Passengers']]-y_train_unnormalized.min())/(y_train_unnormalized.max()-y_train_unnormalized.min())




#%%

plt.figure()
plt.plot(x_train, y_train)
plt.plot(x_test, y_test)

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
        delta = pm.Laplace(f'{name}_delta', mu=0, b = 0.4, shape = (n_time_series, n_changepoints)) # The parameter delta represents the magnitude and direction of the change in growth rate at each changepoint in a piecewise linear model.
        
        # Offset model parameters
        precision_m = pm.Gamma(
            f'precision_{name}_m',
            alpha = prior_alpha,
            beta = prior_beta
            )
        m = pm.Normal(f'{name}_m', mu=0, sigma = 1/np.sqrt(precision_m), shape=n_time_series)
        
        
    return pm.Deterministic(f'{name}_trend',(k + pm.math.dot(A, delta.T))*x[:, None] + m + pm.math.dot(A, (-s * delta).T))


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
            mu = 0,
            sigma = 10,
            shape= (2 * n_fourier_components, dimension)
        )
        
        relative_uncertainty_factor = 1000

        seasonality_period = pm.Gamma(
            f'season_parameter_{name}',
            alpha = relative_uncertainty_factor*seasonality_period_baseline,
            beta = relative_uncertainty_factor
            )
        
        fourier_features = create_fourier_features(
            x=x,
            n_fourier_components=n_fourier_components,
            seasonality_period=seasonality_period
        )
            
    return pm.Deterministic(f'{name}_fourier',pm.math.sum(fourier_features * fourier_coefficients[None,:,:], axis=1))


def pymc_prophet(
        x_train: np.array,
        y_train: np.array,
        seasonality_period_baseline: float,
        n_changepoints: int = 10,
        n_fourier_components: int = 10
        ):
    
    prior_alpha = 2
    prior_beta = 3
    
    with pm.Model() as model:
        n_time_series = y_train.shape[1]
        k_est = (y_train.values[-1]-y_train.values[0])/(x_train[-1]-x_train[0])
        x = pm.Data('x', x_train, dims= 'n_obs')
        y = pm.Data('y', y_train, dims= ['n_obs_target', 'n_time_series'])
        
        linear_term1 = add_linear_term(
                model = model,
                x = x,
                name = 'linear1',
                k_est = k_est,
                n_time_series = n_time_series,
                n_changepoints = n_changepoints,
                prior_alpha = prior_alpha,
                prior_beta = prior_beta
                )
        
        seasonality_individual1 = add_fourier_term(
                model = model,
                x = x,
                n_fourier_components = n_fourier_components,
                name = 'seasonality_individual1',
                dimension = n_time_series,
                seasonality_period_baseline = seasonality_period_baseline,
                prior_alpha = prior_alpha,
                prior_beta =prior_beta
                )
        
        seasonality_shared = add_fourier_term(
                model = model,
                x = x,
                n_fourier_components = n_fourier_components,
                name = 'seasonality_shared',
                dimension = 1,
                seasonality_period_baseline = seasonality_period_baseline,
                prior_alpha = prior_alpha,
                prior_beta =prior_beta
                )
        
        sign_indicator = pm.Bernoulli('sign_indicator', p=0.5, shape = n_time_series)
        prediction = (linear_term1+1)*(1+3*seasonality_individual1)-1 +((2 * sign_indicator - 1))*seasonality_shared
        
        precision_obs = pm.Gamma(
            'precision_obs',
            alpha = prior_alpha,
            beta = prior_beta,
            dims = 'n_time_series'
            )
        
        pm.Normal(
            'obs',
            mu = prediction,
            sigma = 1/np.sqrt(precision_obs),
            observed = y,
            dims = ['n_obs', 'n_time_series']
        )
    return model




#%%


n_fourier_components = 5
n_changepoints = 20

print("n_obs: ", x_train.shape[0])
print("n_time_series: ",y_train.shape[1])
print("n_changepoints: ", n_changepoints)


target = y_train[['Passengers','Mock_Passengers']]


model =  pymc_prophet(
        x_train = x_train,
        y_train = target,
        seasonality_period_baseline = min(significant_periods),
        n_fourier_components = n_fourier_components,
        n_changepoints = n_changepoints
        )


#%%
# Training

with model:
    steps = [pm.NUTS(), pm.HamiltonianMC(), pm.Metropolis()]
    trace = pm.sample(
        tune = 100,
        draws = 500, 
        chains = 1,
        cores = 1,
        step = steps[0]
        #target_accept = 0.9
        )

#%%

for variable in list(trace.posterior.data_vars):
    pm.plot_trace(trace,
                  var_names = [variable])


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


# %% Plot trends and fourier components
print('k values: ', trace.posterior['linear1_k'].mean(('chain','draw')).values)

plt.figure()
plt.plot(x_train,trace.posterior['linear1_trend'].mean(('chain','draw')).values, label = 'linear_trend')
plt.legend()

plt.figure()
plt.plot(x_train,trace.posterior['seasonality_individual1_fourier'].mean(('chain','draw')).values, label = 'seasonality_individual')
plt.legend()

plt.figure()
plt.plot(x_train,trace.posterior['seasonality_shared_fourier'].mean(('chain','draw')).values, label = 'seasonality_shared')
plt.legend()

#%%




#%%
plt.figure()
plt.plot(x_train, (trace.posterior['linear_trend'].mean(('chain','draw')).values+1)*(0+trace.posterior['seasonality_individual_fourier'].mean(('chain','draw')).values)-1, label = 'trend and individual_seasonality')
plt.legend()
