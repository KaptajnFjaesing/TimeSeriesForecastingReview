import pandas as pd

sell_prices_df = pd.read_csv('./data/sell_prices.csv')
train_sales_df = pd.read_csv('./data/sales_train_validation.csv')
calendar_df = pd.read_csv('./data/calendar.csv')
submission_file = pd.read_csv('./data/sample_submission.csv')

#%%

d_cols = [c for c in train_sales_df.columns if 'd_' in c]

# Group by 'state_id' and sum the sales across the specified columns in `d_cols`
sales_sum_df = train_sales_df.groupby(['state_id'])[d_cols].mean().T


# Merge the summed sales data with the calendar DataFrame on the 'd' column and set the index to 'date'
sales_states_df = sales_sum_df.merge(calendar_df.set_index('d')['date'], 
                               left_index=True, right_index=True, 
                               validate="1:1").set_index('date')
# Ensure that the index of your DataFrame is in datetime format
sales_states_df.index = pd.to_datetime(sales_states_df.index)


#%%


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

# Assuming sales_states_df.index is already a datetime index
# Define your threshold date
threshold_date = datetime.datetime(2016, 1, 1)

# Filter the DataFrame to keep only rows with dates greater than the threshold
sales_states__after_data_df = sales_states_df[sales_states_df.index > threshold_date]


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(sales_states__after_data_df.index, sales_states__after_data_df['CA'], label='California')
plt.plot(sales_states__after_data_df.index, sales_states__after_data_df['TX'], label='Texas')
plt.plot(sales_states__after_data_df.index, sales_states__after_data_df['WI'], label='Wisconsin')

# Format the x-axis to show only yearly ticks
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


# Optionally, rotate the x-axis labels for better readability
plt.gcf().autofmt_xdate()

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.title('Sales by State')


#%%

training_split = int(len(sales_states__after_data_df)*0.7)

x_train_unnormalized = sales_states__after_data_df.index[:training_split].astype('int64').values // 10**9
y_train_unnormalized = sales_states__after_data_df.iloc[:training_split]

x_train_mean = x_train_unnormalized.mean()
x_train_std = x_train_unnormalized.std()

y_train_mean = y_train_unnormalized.mean()
y_train_std = y_train_unnormalized.std()

x_train = (x_train_unnormalized-x_train_mean)/x_train_std
y_train = (y_train_unnormalized-y_train_mean)/y_train_std

x_test = (sales_states__after_data_df.index[training_split:].astype('int64').values // 10**9-x_train_mean)/x_train_std
y_test = (sales_states__after_data_df.iloc[training_split:]-y_train_mean)/y_train_std


# %%
import pymc as pm
import numpy as np
import pytensor as pt


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
        y_train = y_train[['CA']],
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

#%%

preds_out_of_sample = posterior_predictive.predictions_constant_data.sortby('x')['x']
model_preds = posterior_predictive.predictions.sortby(preds_out_of_sample)
hdi_values = az.hdi(model_preds)["obs"].transpose("hdi", ...)

plt.figure()
plt.plot(
    preds_out_of_sample,
    model_preds["obs"].mean(("chain", "draw"))
    )
plt.plot(x_train,y_train)

#%%
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