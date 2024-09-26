#%%
import gluonts.torch as gt
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\AWR\Documents\github_repos\TimeSeriesForecastingReview')

from src.utils import normalized_weekly_store_category_household_sales
from src.utils import compute_residuals
from src.utils import CustomBackTransformation

df = normalized_weekly_store_category_household_sales()
df.set_index(df['date'],inplace=True)
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W')
df = df.reindex(full_index)

min_forecast_horizon = 26
max_forecast_horizon = 52
timeseries_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x)] #or ('date' in x) 

df_s = df[timeseries_columns].copy() # make a copy of unscaled data

df[timeseries_columns] = np.log(df[timeseries_columns])
df_0 = df[timeseries_columns].iloc[0,:] # save first datapoint
df = df.diff().bfill()

#%%

model_forecasts_deepar = []
for forecast_horizon in tqdm(range(min_forecast_horizon,max_forecast_horizon+1)):
    # training_data = df[timeseries_columns].iloc[:-forecast_horizon].set_index('date')
    training_data = df[timeseries_columns].iloc[:-forecast_horizon*2]
    validation_data = df[timeseries_columns].iloc[:-forecast_horizon]
    df_1 = np.sum(validation_data)
    y_train_melt = pd.melt(training_data,value_vars=training_data.columns,var_name='ts',value_name='target',ignore_index=False)
    y_validation_melt = pd.melt(validation_data,value_vars=training_data.columns,var_name='ts',value_name='target',ignore_index=False)
    dataset = PandasDataset.from_long_dataframe(y_train_melt,target='target',item_id='ts')
    dataset_validation = PandasDataset.from_long_dataframe(y_validation_melt,target='target',item_id='ts')
    
    gtmodel = gt.DeepAREstimator(
    prediction_length=forecast_horizon,
    context_length=len(training_data)-forecast_horizon,
    freq='W-SUN',
    lr=0.001, #0.000766
    dropout_rate=0.2,
    hidden_size=60, #34
    num_layers=2, #2
    trainer_kwargs={"max_epochs":10}).train(training_data=dataset,
                                            validation_data=dataset_validation) # epochs 15 seem ok

    backtransformation = CustomBackTransformation(constants0=df_0.values,consants1=df_1.values)
    forecasts = list(gtmodel.predict(dataset_validation))
    transformed_forecasts = [backtransformation(forecast, i) for i, forecast in enumerate(forecasts)]

    model_forecasts_deepar.append([pd.Series(forecast.samples.mean(axis=0)) for forecast in transformed_forecasts])
#%%

stacked_deepar = compute_residuals(
         model_forecasts = model_forecasts_deepar,
         test_data = df_s[[x for x in timeseries_columns if 'date' not in x]].iloc[-max_forecast_horizon:],
         min_forecast_horizon = min_forecast_horizon
         )

abs_mean_gradient_training_data = pd.read_pickle('./data/results/abs_mean_gradient_training_data.pkl')
average_abs_residual_deepar = stacked_deepar.abs().groupby(stacked_deepar.index).mean() # averaged over rolls
average_abs_residual_deepar.columns = [x for x in timeseries_columns if 'date' not in x]
MASE_deepar = average_abs_residual_deepar/abs_mean_gradient_training_data

#stacked_deepar.to_pickle(r'.\data\results\stacked_forecasts_deepar.pkl')
# %%
fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(10,15))
i=0
ax = ax.flatten()  # Flatten the 2D array of axes into a 1D array
for forecast, axes in zip(transformed_forecasts,ax):
    axes.plot(df_s[timeseries_columns[i]], color="black",label='True values')
    forecast.plot(ax=axes,name='Forecast')
    axes.tick_params('x',labelrotation=45)
    plt.legend(loc="upper left")
    i=i+1
plt.tight_layout()
plt.show()
# %%
