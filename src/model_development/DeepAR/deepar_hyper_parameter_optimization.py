from bayes_opt import BayesianOptimization
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator

import gluonts.torch as gt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
import pandas as pd
import numpy as np 
import os

os.chdir(r'C:\Users\AWR\Documents\github_repos\TimeSeriesForecastingReview')

from src.utils import normalized_weekly_store_category_household_sales

df = normalized_weekly_store_category_household_sales()
df.set_index(df['date'],inplace=True)
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W')
df = df.reindex(full_index)

unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

df_melt = pd.melt(df,value_vars=unnormalized_column_group,var_name='ts',value_name='target',ignore_index=False)
dataset = PandasDataset.from_long_dataframe(df_melt,target='target',item_id='ts')

training_data, test_gen = split(dataset, offset=-26)
test_data = test_gen.generate_instances(prediction_length=26, windows=1)

def deepar_objective(lr, hidden_size, num_layers, dropout_rate):
    gtmodel = gt.DeepAREstimator(
    prediction_length=26,
    context_length=209-26,
    freq='W-SUN',
    dropout_rate=dropout_rate,
    hidden_size=int(hidden_size),
    num_layers=num_layers,
    lr=lr,
    nonnegative_pred_samples=True,
    trainer_kwargs={"max_epochs":10}).train(training_data)

    forecasts = list(gtmodel.predict(test_data.input))

    mae_array = np.zeros(10)
    i=0
    for forecast, instance in zip(forecasts,test_data):
        mae_array[i]=np.mean(np.abs(instance[1]['target']-forecast.median))
        i=i+1

    return -np.mean(mae_array)

param_bounds = {
    'lr': (1e-4, 1e-1),
    'hidden_size': (20, 80),
    'num_layers': (1, 5),
    'dropout_rate': (0.05, 0.4)
}

optimizer = BayesianOptimization(
    f=deepar_objective,
    pbounds=param_bounds)

optimizer.maximize(init_points=5,n_iter=25)

print(optimizer.max)

"""
df_melt = pd.melt(df,value_vars=y_train.columns,var_name='ts',value_name='target',ignore_index=False)
dataset = PandasDataset.from_long_dataframe(df_melt,target='target',item_id='ts')

training_data, test_gen = split(dataset, offset=-26)
test_data = test_gen.generate_instances(prediction_length=26, windows=1)

gtmodel = gt.DeepAREstimator(
    prediction_length=26,
    context_length=209-26,
    freq='W-SUN',
    lr=0.000766,
    dropout_rate=0.105,
    hidden_size=34,
    num_layers=2,
    nonnegative_pred_samples=True,
    trainer_kwargs={"max_epochs":15}).train(training_data)

forecasts = list(gtmodel.predict(test_data.input))

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(10,15))
i=0
ax = ax.flatten()  # Flatten the 2D array of axes into a 1D array
for forecast, axes in zip(forecasts,ax):
    axes.plot(df[y_train.columns[i]], color="black",label='True values')
    forecast.plot(ax=axes,name='Forecast')
    axes.tick_params('x',labelrotation=45)
    plt.legend(loc="upper left")
    i=i+1

plt.tight_layout()
plt.show()
"""
        