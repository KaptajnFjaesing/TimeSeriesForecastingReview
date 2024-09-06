# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:22:27 2024

@author: petersen.jonas
"""


from src.functions import (
    normalized_weekly_store_category_household_sales,
    load_m5_weekly_store_category_sales_data
    )
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor as pt


df = normalized_weekly_store_category_household_sales()
_,weekly_store_category_household_sales,_ = load_m5_weekly_store_category_sales_data()


n_weeks = 52
normalized_column_group = [x for x in df.columns if '_normalized' in x ]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]


training_data = df.iloc[:-n_weeks]
test_data = df.iloc[-n_weeks:]

future_mean_model = test_data[normalized_column_group].mean(axis = 1)


# %%
x_train = (training_data['date'].astype('int64')//10**9 - (training_data['date'].astype('int64')//10**9).min())/((training_data['date'].astype('int64')//10**9).max() - (training_data['date'].astype('int64')//10**9).min())
y_train = (training_data[unnormalized_column_group]-training_data[unnormalized_column_group].min())/(training_data[unnormalized_column_group].max()-training_data[unnormalized_column_group].min())

x_test = (test_data['date'].astype('int64')//10**9 - (training_data['date'].astype('int64')//10**9).min())/((training_data['date'].astype('int64')//10**9).max() - (training_data['date'].astype('int64')//10**9).min())
y_test = (test_data[unnormalized_column_group]-training_data[unnormalized_column_group].min())/(training_data[unnormalized_column_group].max()-training_data[unnormalized_column_group].min())

x_total = (df['date'].astype('int64')//10**9 - (training_data['date'].astype('int64')//10**9).min())/((training_data['date'].astype('int64')//10**9).max() - (training_data['date'].astype('int64')//10**9).min())
y_total = (df[unnormalized_column_group]-training_data[unnormalized_column_group].min())/(training_data[unnormalized_column_group].max()-training_data[unnormalized_column_group].min())



# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil(y_test.shape[1] / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(y_test.shape[1]):
    ax = axs[i]  # Get the correct subplot
    ax.plot(x_train, y_train[y_train.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(x_test, y_test[y_test.columns[i]], color = 'tab:blue', label='Test Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure
# %%
def determine_significant_periods(
        series: np.array,
        x_train: np.array,
        threshold: float
        ) -> tuple:
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


    # Find components that are more than K fraction of the maximum
    significant_indices = np.where(positive_magnitudes >= threshold * max_magnitude)[0]
    significant_frequencies = positive_freqs[significant_indices]
    significant_periods = 1 / significant_frequencies
    return dominant_period, significant_periods


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
        m_est:float,
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
        m = pm.Normal(f'{name}_m', mu=m_est, sigma = 1/np.sqrt(precision_m), shape=n_time_series)
        
        
    return pm.Deterministic(f'{name}_trend',(k + pm.math.dot(A, delta.T))*x[:, None] + m + pm.math.dot(A, (-s * delta).T))


def add_fourier_term(
        model,
        x: np.array,
        n_fourier_components: int,
        name: str,
        dimension: int,
        seasonality_period_baseline: float
        ):
    with model:
        fourier_coefficients = pm.Normal(
            f'fourier_coefficients_{name}',
            mu = 0,
            sigma = 1,
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


def sorcerer(
        x_train: np.array,
        y_train: np.array,
        seasonality_period_baseline: float,
        n_groups: int = 3,
        n_changepoints: int = 10,
        n_fourier_components: int = 10
        ):
    
    prior_alpha = 2
    prior_beta = 3
    
    
    n_time_series = y_train.shape[1]
    n_obs = y_train.shape[0]
    k_est = (y_train.values[-1]-y_train.values[0])/(x_train[-1]-x_train[0])
    m_est = k_est = y_train.values[0]
    
    with pm.Model() as model:
        
        
        x = pm.Data('x', x_train, dims= 'n_obs')
        y = pm.Data('y', y_train, dims= ['n_obs_target', 'n_time_series'])
        
        linear_term1 = add_linear_term(
                model = model,
                x = x,
                name = 'linear1',
                k_est = k_est,
                m_est = m_est,
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
                seasonality_period_baseline = seasonality_period_baseline
                )

        seasonality_shared = add_fourier_term(
                model = model,
                x = x,
                n_fourier_components = n_fourier_components,
                name = 'seasonality_shared',
                dimension = n_groups-1,
                seasonality_period_baseline = seasonality_period_baseline
                )
        
        all_models = pm.math.concatenate([
            pm.math.zeros((n_obs, 1)),
            seasonality_shared]
            , axis=1)  # Shape: (n_groups+1, n_obs)
        model_probs = pm.Dirichlet('model_probs', a=np.ones(n_groups), shape=(n_time_series, n_groups))
        chosen_model_index = pm.Categorical('chosen_model_index', p=model_probs, shape=n_time_series)
        selected_models = pm.Deterministic(
            'selected_models', 
            all_models[:, chosen_model_index]
        )
        
        prediction = (
            linear_term1+
            seasonality_individual1+
            selected_models
            )

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
"""
TO THIS POINT:
    I am trying to get the concatenation in 
    all_models = pm.math.concatenate([pm.math.zeros((n_obs, 1)), normal_models], axis=1) 
    to work.

"""

with pm.Model() as model:
    n_groups = 4
    n_obs = 52
    n_time_series = 11
    normal_models = pm.Normal(
        'obs',
        mu = 0,
        sigma = 1,
        shape = (n_obs,n_groups-1)
    )
    
    zero_model = np.zeros(n_obs)  # Zero model with the same shape as the observations

    all_models = pm.math.concatenate([pm.math.zeros((n_obs, 1)), normal_models], axis=1)  # Shape: (n_groups+1, n_obs)
    
    model_probs = pm.Dirichlet('model_probs', a=np.ones(n_groups), shape=(n_time_series, n_groups))
    print(model_probs.shape.eval())
    chosen_model_index = pm.Categorical('chosen_model_index', p=model_probs, shape=n_time_series)
        
    print(all_models[:,chosen_model_index].shape.eval())
    selected_models = pm.Deterministic(
        'selected_models', 
        all_models[:, chosen_model_index]
    )
# %%
(dominant_period, significant_periods) = determine_significant_periods(
        series = y_train.values[:,0],
        x_train = x_train.values,
        threshold = 0.5
        )

n_fourier_components = 10
n_changepoints = 20

print("n_obs: ", x_train.shape[0])
print("n_time_series: ",y_train.shape[1])
print("n_changepoints: ", n_changepoints)



model =  sorcerer(
        x_train = x_train.values,
        y_train = y_train,
        seasonality_period_baseline = significant_periods[2],
        n_fourier_components = n_fourier_components,
        n_changepoints = n_changepoints
        )


#%%
# Training

with model:
    steps = [pm.NUTS(), pm.HamiltonianMC(), pm.Metropolis()]
    trace = pm.sample(
        tune = 100,
        draws = 200, 
        chains = 1,
        cores = 1,
        step = steps[0]
        )




#%%

for variable in list(trace.posterior.data_vars):
    pm.plot_trace(trace,
                  var_names = [variable])


#%%


import arviz as az

X = x_test

with model:
    pm.set_data({'x':X})
    posterior_predictive = pm.sample_posterior_predictive(trace = trace, predictions=True)

preds_out_of_sample = posterior_predictive.predictions_constant_data.sortby('x')['x']
model_preds = posterior_predictive.predictions.sortby(preds_out_of_sample)

hdi_values = az.hdi(model_preds)["obs"].transpose("hdi", ...)

# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil(y_test.shape[1] / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(y_test.shape[1]):
    ax = axs[i]  # Get the correct subplot
    ax.plot(x_total, y_total[y_total.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(preds_out_of_sample, (model_preds["obs"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    ax.fill_between(
        preds_out_of_sample.values,
        hdi_values[0].values[:,i],  # lower bound of the HDI
        hdi_values[1].values[:,i],  # upper bound of the HDI
        color= 'blue',   # color of the shaded region
        alpha=0.4,      # transparency level of the shaded region
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure


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


# %%

def test_r_squared_absolute(
        test_data,
        model_predictions
        ):
    n_weeks = len(model_predictions)
    unnormalized_column_group = [x for x in test_data.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
    n_groups = len(unnormalized_column_group)
    ss_res_raw = np.ones((n_weeks,n_groups))
    for week in range(1,n_weeks+1):
        df_test_week = test_data[test_data['week'] == week][unnormalized_column_group]
        if len(df_test_week) == 0:
            print("No test data")
            return np.nan, np.nan
        if len(model_predictions) == 0:
            print("No profile data")
            return np.nan, np.nan
        ss_res_raw[week-1] = ((df_test_week - model_predictions[week-1])**2).values[0,:]
        
    ss_res = ss_res_raw.mean(axis = 1)

    return ss_res



# %% de-normalizing predictions

m_preds1 = (model_preds["obs"].mean(("chain", "draw")).T).values
v1 = (training_data[unnormalized_column_group].max()-training_data[unnormalized_column_group].min()).values[:,np.newaxis]
v2 = training_data[unnormalized_column_group].min().values[:,np.newaxis]

m_preds = (m_preds1*v1+v2)

# %%

hest = test_r_squared_absolute(
        test_data = test_data,
        model_predictions = m_preds.T
        )

ko = np.array([ 324066.27020221,  353861.59978992,  359154.8869314 ,
        461383.36302469,  499535.43893675,  661811.9474528 ,
        608567.4498722 ,  616030.18662155,  743620.06973728,
        527215.29526805,  570638.21355619,  505842.41322868,
        402848.84765924,  349714.4173384 ,  480842.06805023,
        556501.07213472,  375756.68243674,  389934.08749543,
        298323.37408965,  594219.78328139,  499853.61571022,
        488511.26398265,  515151.57002577,  443897.32326999,
        481654.81253097,  552350.84163434,  519299.14397887,
        500198.8298041 ,  570600.80624516,  574554.70176082,
        548735.96112816,  563189.16603876,  750771.2125082 ,
        842154.11308326,  649789.35308349,  837039.25597015,
        664096.82995281,  518863.05281273,  594115.68672087,
        747065.00945079,  716956.72893107,  704981.45620959,
        695077.32461338,  629963.20618933,  810912.18112303,
        753211.15295953,  823199.94535604,  804856.02009401,
       1050495.33897366,  965715.70486884, 1108543.60467126,
       1423135.35538981])

r_squared = 1-hest/ko

plt.figure()
plt.plot(hest, label = "Sorcerer model")
plt.plot(ko, label = "Future mean model")
plt.legend()

print(r_squared.mean().round(2),"+-",r_squared.std().round(2))
