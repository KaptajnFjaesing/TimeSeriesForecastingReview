from src.functions import load_passengers_data
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pytensor as pt

df_passengers = load_passengers_data()  # Load the data

# Generate mock time series
np.random.seed(42)  # For reproducibility
seasonality = 15 * np.sin(2 * np.pi * np.arange(len(df_passengers)) / 12)
trend = 80 + 1.2 * np.arange(len(df_passengers))
noise = np.random.normal(0, 10, len(df_passengers))
df_passengers['Mock_Passengers'] = (seasonality + trend + noise).astype(int)


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




#%%
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

        season_parameter = pm.Normal(f'season_parameter_{name}', mu=0, sigma=5, shape = dimension)
        seasonality_period = seasonality_period_baseline * pm.math.exp(season_parameter)
        
        fourier_features = create_fourier_features(
            x=x,
            n_fourier_components=n_fourier_components,
            seasonality_period=seasonality_period
        )
            
    output = pt.tensor.sum(fourier_features * fourier_coefficients[None,:,:], axis=1)
    print("fourier_term_dimensions: ",output.shape.eval())
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
    
    prior_alpha = 1
    prior_beta = 1
    with pm.Model() as model:
        # Data variables
        n_time_series = y_train.shape[1]
        k_est = (y_train.values[-1][0]-y_train.values[0][0])/(x_train[-1]-x_train[0])
        x = pm.Data('x', x_train, dims= ['n_obs'])
        y = pm.Data('y', y_train, dims= ['n_obs_target', 'n_time_series'])
    
        s = pt.tensor.linspace(0, pm.math.max(x), n_changepoints+2)[1:-1]
        A = (x[:, None] > s)*1.
    
        # Growth model parameters
        precision_k = pm.Gamma(
            'precision_k',
            alpha = prior_alpha,
            beta = prior_beta
            )
        k = pm.Normal(
            'k',
            mu = k_est,
            sigma = 1/np.sqrt(precision_k),
            shape = n_time_series
            )  # Shape matches time series
        delta = pm.Laplace('delta', mu=0, b=1, shape = (n_time_series, n_changepoints)) # The parameter delta represents the magnitude and direction of the change in growth rate at each changepoint in a piecewise linear model.
        growth = k + pm.math.dot(A, delta.T)  # Shape: (n_obs, n_time_series)
        
        # Offset model parameters
        precision_m = pm.Gamma(
            'precision_m',
            alpha = prior_alpha,
            beta = prior_beta
            )
        m = pm.Normal('m', mu=0, sigma = 1/np.sqrt(precision_m), shape=n_time_series)
        
        gamma = pm.Deterministic('gamma', -s * delta) # (n_time_series, n_changepoints)
        offset = m + pm.math.dot(A, gamma.T)  # Shape: (n_obs, n_time_series)
    
        # trend
        trend = growth*x[:, None] + offset  # Shape: (n_obs, n_time_series)
        
        # shared Seasonality model
        seasonality_shared = add_fourier_term(
                model = model,
                x = x,
                n_fourier_components = n_fourier_components_shared,
                name = "shared",
                dimension = 1,
                seasonality_period_baseline = seasonality_period_baseline_shared,
                prior_alpha = prior_alpha,
                prior_beta = prior_beta
                ) # (n_obs,n_time_series)
        

        seasonality_individual = add_fourier_term(
                model = model,
                x = x,
                n_fourier_components = n_fourier_components_individual,
                name = "individual",
                dimension = n_time_series,
                seasonality_period_baseline = seasonality_period_baseline_individual,
                prior_alpha = prior_alpha,
                prior_beta = prior_beta
                ) # (n_obs, n_time_series)
        
        prediction = pm.Deterministic('prediction',trend*(1+seasonality_individual)+seasonality_shared) # (n_obs, n_time_series)
        
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


n_fourier_components_shared = 10
seasonality_period_baseline_shared = min(significant_periods)

n_fourier_components_individual = 10
seasonality_period_baseline_individual = min(significant_periods)

n_changepoints = 20

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


#%%
# Training


with model:
    steps = [pm.NUTS, pm.HamiltonianMC(), pm.Metropolis()]
    trace = pm.sample(
        tune = 100,
        draws = 500, 
        chains = 1,
        cores = 1,
        step = steps[1]
        #target_accept = 0.9
        )

#%%



pm.plot_trace(
    trace,
    var_names = [
        'k',
        'm',
        'delta',
        'fourier_coefficients_shared'
        ]
    )


pm.plot_trace(
    trace,
    var_names = [
        'season_parameter_shared',
        'fourier_coefficients_individual',
        'season_parameter_individual',
        'precision_obs'
        ]
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


for i in range(y_train.shape[1]):
    plt.figure()
    plt.title(y_train.columns[i])
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

    plt.plot(x_train,y_train[y_train.columns[i]], color = 'red')
    plt.plot(x_test,y_test[y_test.columns[i]], color = 'green')





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



seasonality_period_baseline = seasonality_period_baseline_shared
n_fourier_components = n_fourier_components_shared


with pm.Model() as model:
    # Data variables
    n_time_series = y_train.shape[1]
    dimension = 1
    
    x = pm.Data('x', x_train, dims= ['n_obs'])
    
    
    fourier_coefficients = pm.Normal(
        f'fourier_coefficients',
        mu=0,
        sigma=5,
        shape = (2*n_fourier_components,n_time_series)
    )
    
    print(x.shape.eval())
    season_parameter = pm.Normal('season_parameter', mu=0, sigma=5, shape = dimension)
    seasonality_period = seasonality_period_baseline * pm.math.exp([0])
    
    fourier_features = create_fourier_features(
        x=x,
        n_fourier_components=n_fourier_components,
        seasonality_period=seasonality_period
    )
    
    print(fourier_coefficients[None,:,None].shape.eval())
    output = pt.tensor.sum(fourier_features * fourier_coefficients[None,:,:], axis=1) # (n_obs, n_time_series)
    #output1 = pt.tensordot(fourier_features, fourier_coefficients[None,:,:])
    
    #print(fourier_features.eval())
    print("Fourier features shape: ",fourier_features.shape.eval())
    print("Fourier coefficients shape: ",fourier_coefficients.shape.eval())
    print("Output shape: ",output.shape.eval())
    
    st_fourier_features = fourier_features.eval()
    st_fourier_coefficients = fourier_coefficients.eval()
    st_output = output.eval()


seasonality_period_baseline = np.array(seasonality_period_baseline_shared)
frequency_component = 2 * np.pi * (np.arange(n_fourier_components)+1) * x_train[:, None] # (n_obs, n_fourier_components)
t = frequency_component / seasonality_period_baseline_shared # (n_obs, n_fourier_components, n_time_series)
fourier_features = np.concatenate((np.cos(t), np.sin(t)), axis = 1)  # (n_obs, 2*n_fourier_components, n_time_series)

print(fourier_features.shape)

# %%


print(np.dot(st_fourier_features[:,:,0],st_fourier_coefficients))