#%% Import section
from src.functions import load_passengers_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df_passengers = load_passengers_data()  # Load the data

#%% Statespace model: Unobserved components model (structured time series)

forecast_horizon = 26
x_train = df_passengers['Passengers'].values[0:len(df_passengers)-forecast_horizon]
struobs = sm.tsa.UnobservedComponents(x_train,
                                      level=True,
                                      trend=True,
                                      autoregressive=1,
                                      stochastic_level=True,
                                      stochastic_trend=True,
                                      freq_seasonal=[{'period':12,
                                                      'harmonics':4}],
                                      irregular=True
                                      )

mlefit = struobs.fit()
mle_df = mlefit.get_forecast(forecast_horizon).summary_frame(alpha=0.05)
#%% 
mlefit.summary()
#%%
mlefit.plot_components(figsize=(16,12))
plt.show()
#%%
plt.plot(df_passengers['Passengers'],label='Actual')
plt.plot(np.arange(len(df_passengers)-forecast_horizon,len(df_passengers)),mle_df['mean'],label='MLE_mean_forecast')
plt.fill_between(np.arange(len(df_passengers)-forecast_horizon,len(df_passengers)),mle_df['mean_ci_lower'],mle_df['mean_ci_upper'],color='blue',alpha=0.2,label='MLE_forecast_CI')
plt.xlabel('Time (months)')
plt.ylabel('Nr. of passengers')
plt.legend()
plt.show()
