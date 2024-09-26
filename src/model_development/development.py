# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:37:20 2024

@author: petersen.jonas
"""

import pandas as pd
import datetime
from darts import TimeSeries
from darts.utils.utils import ModelMode, SeasonalityMode

def load_m5_data() -> tuple[pd.DataFrame]:
    sell_prices_df = pd.read_csv('./data/sell_prices.csv')
    train_sales_df = pd.read_csv('./data/sales_train_validation.csv')
    calendar_df = pd.read_csv('./data/calendar.csv')
    submission_file = pd.read_csv('./data/sample_submission.csv')
    return (
        sell_prices_df,
        train_sales_df,
        calendar_df,
        submission_file
        )

def load_m5_weekly_store_category_sales_data():
    
    (sell_prices_df, train_sales_df, calendar_df, submission_file) = load_m5_data()
    
    threshold_date = datetime.datetime(2011, 1, 1)
    
    d_cols = [c for c in train_sales_df.columns if 'd_' in c]
    
    train_sales_df['store_cat'] = train_sales_df['store_id'].astype(str) + '_' + train_sales_df['cat_id'].astype(str)
    # Group by 'state_id' and sum the sales across the specified columns in `d_cols`
    sales_sum_df = train_sales_df.groupby(['store_cat'])[d_cols].sum().T
    
    # Merge the summed sales data with the calendar DataFrame on the 'd' column and set the index to 'date'
    store_category_sales = sales_sum_df.merge(calendar_df.set_index('d')['date'], 
                                   left_index=True, right_index=True, 
                                   validate="1:1").set_index('date')
    # Ensure that the index of your DataFrame is in datetime format
    store_category_sales.index = pd.to_datetime(store_category_sales.index)
    weekly_store_category_sales = store_category_sales[store_category_sales.index > threshold_date].resample('W-MON', closed = "left", label = "left").sum().iloc[1:]
    
    food_columns = [x for x in weekly_store_category_sales.columns if 'FOOD' in x]
    household_columns = [x for x in weekly_store_category_sales.columns if 'HOUSEHOLD' in x]
    hobbies_columns = [x for x in weekly_store_category_sales.columns if 'HOBBIES' in x]
    
    weekly_store_category_food_sales = weekly_store_category_sales[food_columns]
    weekly_store_category_household_sales = weekly_store_category_sales[household_columns]
    weekly_store_category_hobbies_sales = weekly_store_category_sales[hobbies_columns]
    
    return (
        weekly_store_category_food_sales.reset_index(),
        weekly_store_category_household_sales.reset_index(),
        weekly_store_category_hobbies_sales.reset_index()
        )


_,df,_ = load_m5_weekly_store_category_sales_data()

# %%
from tqdm import tqdm
import darts.models as dm

def generate_exponential_smooothing_darts_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int
        ):
    time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
    model_forecasts = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        exp_model = dm.ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.MULTIPLICATIVE)
        series = []
        for time_series in time_series_columns:
            exp_model.fit(TimeSeries.from_dataframe(df.iloc[:-fh], 'date', time_series))
            exp_predict = exp_model.predict(forecast_horizon)
            series.append(pd.Series(exp_predict.values()))
        model_forecasts.append(series)
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle("./data/results/stacked_forecasts_exp_smoothing_darts.pkl")
    print("generate_exponential_smooothing_darts_stacked_forecast completed")



def compute_residuals(model_forecasts, test_data, min_forecast_horizon):
    metrics_list = []
    for forecasts in model_forecasts:
        errors_df = pd.DataFrame(columns=test_data.columns)
        for i, column in enumerate(test_data.columns):
            forecast_horizon = len(forecasts[i].values)
            errors_df[column] = test_data[column].values[-forecast_horizon:] - forecasts[i].values
        truncated_errors_df = errors_df.iloc[:min_forecast_horizon].reset_index(drop=True)        
        metrics_list.append(truncated_errors_df)
    return pd.concat(metrics_list, axis=0)


forecast_horizon = 26
simulated_number_of_forecasts = 50
number_of_weeks_in_a_year = 52.1429
context_length = 30

generate_exponential_smooothing_darts_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts 
        )

# %%


def generate_tide_darts_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        model_config: dict
        ):
    time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
    model_forecasts_naive = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        tide_model = dm.TiDEModel(**model_config)
        tide_model.fit(TimeSeries.from_dataframe(df.iloc[:-fh], 'date'))
        tide_predict = tide_model.predict(forecast_horizon)    
        model_forecasts_naive.append([pd.Series(tide_predict.values()[:,i]) for i in range(len(time_series_columns))])
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts_naive,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle("./data/results/stacked_forecasts_tide_darts.pkl")
    print("generate_tide_darts_stacked_forecast completed")




tide_darts_config = {
    'input_chunk_length': 64,  # Input sequence length
    'output_chunk_length': forecast_horizon,  # Forecast horizon
    'hidden_size': 128,  # Size of hidden layers
    'n_epochs': 100,  # More epochs for better training
    'dropout': 0.1,  # Regularization to prevent overfitting
    'optimizer_kwargs': {'lr': 0.001},  # Learning rate for optimizer
    'random_state': 42,  # For reproducibility
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'decoder_output_dim': 4
    }


generate_tide_darts_stacked_forecast(
        df = df,
        forecast_horizon = forecast_horizon,
        simulated_number_of_forecasts = simulated_number_of_forecasts,
        model_config = tide_darts_config
        )