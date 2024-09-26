#%%
from bayes_opt import BayesianOptimization
from gluonts.transform import MapTransformation
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import matplotlib.pyplot as plt
import gluonts.torch as gt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
import pandas as pd
import numpy as np 
import os

os.chdir(r'C:\Users\AWR\Documents\github_repos\TimeSeriesForecastingReview')

from src.utils import normalized_weekly_store_category_household_sales

# %%
class CustomBackTransformation:
    def __init__(self, constants0,consants1):
        self.constants0 = constants0
        self.constants1 = consants1

    def __call__(self, forecast, index):
        # Apply the transformation with the constant for the specific forecast index
        forecast.samples = np.exp(np.cumsum(forecast.samples, axis=1) + self.constants0[index] + self.constants1[index])
        return forecast
# %%

df = normalized_weekly_store_category_household_sales()
df.set_index(df['date'],inplace=True)
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W')
df = df.reindex(full_index)


timeseries_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x)] 

#df[timeseries_columns] = df[timeseries_columns].apply(lambda x: np.log(x) if x.name != 'date' else x)
df[timeseries_columns] = np.log(df[timeseries_columns])
df_0 = df[timeseries_columns].iloc[0,:] # save first datapoint
df = df.diff().bfill()

df_melt = pd.melt(df,value_vars=timeseries_columns,var_name='ts',value_name='target',ignore_index=False)
df_melt['sine_year']=np.sin(2*np.pi*df_melt.index.to_series().dt.isocalendar().week.values/52)

dataset = PandasDataset.from_long_dataframe(
    df_melt,
    target='target',
    item_id='ts',
    #feat_dynamic_real=['sine_year']
    )

#training_data, test_gen = split(dataset, offset=-39*2)
#test_data = test_gen.generate_instances(prediction_length=39, windows=1)
forecast_horizon = 39

training_data = df[timeseries_columns].iloc[:-forecast_horizon*2]
test_data = df[timeseries_columns].iloc[:-forecast_horizon]
y_train_melt = pd.melt(training_data,value_vars=training_data.columns,var_name='ts',value_name='target',ignore_index=False)
y_validation_melt = pd.melt(test_data,value_vars=training_data.columns,var_name='ts',value_name='target',ignore_index=False)
dataset = PandasDataset.from_long_dataframe(y_train_melt,target='target',item_id='ts')
dataset_validation = PandasDataset.from_long_dataframe(y_validation_melt,target='target',item_id='ts')

def deepar_objective(lr, hidden_size, num_layers, dropout_rate):
    gtmodel = gt.DeepAREstimator(
    prediction_length=39,
    context_length=len(training_data)-forecast_horizon,
    freq='W-SUN',
    dropout_rate=dropout_rate,
    hidden_size=int(hidden_size),
    num_layers=num_layers,
    lr=lr,
    patience=4,
    trainer_kwargs={"max_epochs":15}).train(training_data=dataset,
                                            validation_data=dataset_validation)

    forecasts = list(gtmodel.predict(dataset_validation))

    mae_array = np.zeros(10)
    i=0
    for forecast, col in zip(forecasts,timeseries_columns):
        mae_array[i]=np.mean(np.abs(df[col].iloc[-forecast_horizon:]-forecast.median))
        i=i+1

    return -np.mean(mae_array)

param_bounds = {
    'lr': (1e-4, 1e-2),
    'hidden_size': (20, 80),
    'num_layers': (1, 5),
    'dropout_rate': (0.01, 0.3)
}

optimizer = BayesianOptimization(
    f=deepar_objective,
    pbounds=param_bounds)

#optimizer.maximize(init_points=5,n_iter=25)

#print(optimizer.max)
#%%
#training_data, test_gen = split(dataset, offset=-39*2)
#test_data = test_gen.generate_instances(prediction_length=39, windows=1)

gtmodel = gt.DeepAREstimator(
    prediction_length=forecast_horizon,
    context_length=len(training_data)-forecast_horizon,
    freq='W-SUN',
    lr=0.001307,#0.001,
    dropout_rate=0.17,
    hidden_size=75,#32
    num_layers=2,#2
    patience=4,
    trainer_kwargs={"max_epochs":15}).train(training_data=dataset,
                                            validation_data=dataset_validation)
#%%
forecasts = list(gtmodel.predict(dataset_validation))
#%%
fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(10,15))
i=0
ax = ax.flatten()  # Flatten the 2D array of axes into a 1D array
for forecast, axes in zip(forecasts,ax):
    axes.plot(df[timeseries_columns[i]], color="black",label='True values')
    forecast.plot(ax=axes,name='Forecast')
    axes.tick_params('x',labelrotation=45)
    plt.legend(loc="upper left")
    i=i+1

plt.tight_layout()
plt.show()
#%%
forecasts = list(gtmodel.predict(dataset_validation))
df_1 = np.sum(df[timeseries_columns],axis=0)
back_transformation = CustomBackTransformation(constants0=df_0.values,consants1=df_1.values)

transformed_forecasts = [back_transformation(forecast, i) for i, forecast in enumerate(forecasts)]

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(10,15))
i=0
ax = ax.flatten()  # Flatten the 2D array of axes into a 1D array
for transformed_forecast, axes in zip(transformed_forecasts,ax):
    axes.plot(np.exp(np.cumsum(df[timeseries_columns[i]])+df_0.iloc[i]), color="black",label='True values')
    transformed_forecast.plot(ax=axes,name='Forecast')
    axes.tick_params('x',labelrotation=45)
    plt.legend(loc="upper left")
    i=i+1
plt.ylim((2000,9000))
plt.tight_layout()
plt.show()

# %%
forecasts = list(gtmodel.predict(test_data.input))
fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(10,15))
i=0
ax = ax.flatten()  # Flatten the 2D array of axes into a 1D array
for forecast, axes in zip(forecasts,ax):
    axes.plot(np.exp(np.cumsum(df[timeseries_columns[i]])+df_0.iloc[i]), color="black",label='True values')
    tfor = np.exp(np.cumsum(np.mean(forecast.samples,axis=0))+df_0.iloc[i]+df_1.iloc[i])
    axes.plot(df.iloc[209-52:,:].index,tfor, color="blue",label='forecast')
    axes.tick_params('x',labelrotation=45)
    plt.legend(loc="upper left")
    i=i+1

plt.tight_layout()
plt.show()
# %%
