import gluonts.torch as gt
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
from tqdm import tqdm
import os

os.chdir(r'C:\Users\AWR\Documents\github_repos\TimeSeriesForecastingReview')

from src.utils import normalized_weekly_store_category_household_sales
from src.utils import compute_residuals

df = normalized_weekly_store_category_household_sales()
df.set_index(df['date'],inplace=True)
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W')
df = df.reindex(full_index)

min_forecast_horizon = 26
max_forecast_horizon = 52
timeseries_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)] 

model_forecasts_deepar = []
for forecast_horizon in tqdm(range(min_forecast_horizon,max_forecast_horizon+1)):
    training_data = df[timeseries_columns].iloc[:-forecast_horizon].set_index('date')
    y_train_melt = pd.melt(training_data,value_vars=training_data.columns,var_name='ts',value_name='target',ignore_index=False)
    dataset = PandasDataset.from_long_dataframe(y_train_melt,target='target',item_id='ts')
    
    gtmodel = gt.DeepAREstimator(
    prediction_length=forecast_horizon,
    context_length=len(training_data),
    freq='W-SUN',
    lr=0.000766,
    dropout_rate=0.105,
    hidden_size=34,
    num_layers=2,
    nonnegative_pred_samples=True,
    trainer_kwargs={"max_epochs":10}).train(dataset)

    model_forecasts_deepar.append([pd.Series(forecast.samples.mean(axis=0)) for forecast in list(gtmodel.predict(dataset))])


stacked_deepar = compute_residuals(
         model_forecasts = model_forecasts_deepar,
         test_data = df[[x for x in timeseries_columns if 'date' not in x]].iloc[-max_forecast_horizon:],
         min_forecast_horizon = min_forecast_horizon
         )

abs_mean_gradient_training_data = pd.read_pickle('./data/results/abs_mean_gradient_training_data.pkl')
average_abs_residual_deepar = stacked_deepar.abs().groupby(stacked_deepar.index).mean() # averaged over rolls
average_abs_residual_deepar.columns = [x for x in timeseries_columns if 'date' not in x]
MASE_deepar = average_abs_residual_deepar/abs_mean_gradient_training_data

stacked_deepar.to_pickle(r'.\data\results\stacked_forecasts_deepar.pkl')