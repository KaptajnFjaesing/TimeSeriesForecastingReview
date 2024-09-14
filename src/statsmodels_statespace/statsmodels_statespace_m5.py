# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:53:07 2024

@author: petersen.jonas
"""

from src.load_data import normalized_weekly_store_category_household_sales
import statsmodels.api as sm

df = normalized_weekly_store_category_household_sales()

#%%
from tqdm import tqdm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress specific warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


"""
The idea is to split data into test and training in terms of the forecast
horizon viz

    training:   :-forecast_horizon
    test:       -forecast_horizon:
    
then loop over forecast horizons in the range 

    (min_forecast_horizon, max_in_test_data).

For each forecast, pick the min_forecast_horizon only. This will produce a set
of forecasts of length min_forecast_horizon, which is rolled over test data.
This method is just an easy way to produce k-fold cross validation with k equal
to max_in_test_data-min_forecast_horizon.

Afterwars, the mean and standard deviation over each forecast step is taken.
This produces data than can be plotted as a shaded region for each time series,
that represent the mean residual and the standard deviation of residuals.

"""

seasonality_period = 52
harmonics = 10
autoregressive = 2
min_forecast_horizon = 26
max_forecast_horizon = 52
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

model_forecasts = []
for forecast_horizon in tqdm(range(min_forecast_horizon,max_forecast_horizon+1)):
    training_data = df.iloc[:-forecast_horizon].reset_index()
    y_train_min = training_data[unnormalized_column_group].min()
    y_train_max = training_data[unnormalized_column_group].max()
    y_train = (training_data[unnormalized_column_group]-y_train_min)/(y_train_max-y_train_min)
    model_denormalized = []
    for i in range(len(y_train.columns)):
        struobs = sm.tsa.UnobservedComponents(y_train[y_train.columns[i]],
                                              level=True,
                                              trend=True,
                                              autoregressive=autoregressive,
                                              stochastic_level=True,
                                              stochastic_trend=True,
                                              freq_seasonal=[{'period':seasonality_period,
                                                              'harmonics':harmonics}],
                                              irregular=True
                                              )
        mlefit = struobs.fit(disp=0)
        model_results = mlefit.get_forecast(forecast_horizon).summary_frame(alpha=0.05)
        model_denormalized.append(model_results['mean']*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]])

    model_forecasts.append(model_denormalized)

# %%

from src.utils import compute_residuals

test_data = df.iloc[-max_forecast_horizon:].reset_index(drop = True)

stacked = compute_residuals(
         model_forecasts = model_forecasts,
         test_data = test_data[unnormalized_column_group],
         min_forecast_horizon = min_forecast_horizon
         )   

stacked.to_pickle('./data/results/stacked_forecasts_statespace.pkl')



