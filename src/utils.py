# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:25:03 2024

@author: petersen.jonas
"""
import pandas as pd
import datetime

def compute_metrics(model_forecasts, test_data, min_forecast_horizon):
    """
    Compute the forecast error metrics for multiple models.

    Parameters:
    - model_forecasts: List of forecasted values from different models. Each element is a list of forecasts for each column.
    - test_data: DataFrame containing the actual values for the forecast period.
    - min_forecast_horizon: The minimum forecast horizon to consider for each model.

    Returns:
    - List of DataFrames, each containing the forecast errors for a model.
    """
    metrics_list = []

    # Iterate over each model's forecasts
    for forecasts in model_forecasts:
        # Initialize an empty DataFrame to store forecast errors for the current model
        errors_df = pd.DataFrame(columns=test_data.columns)
        
        # Calculate forecast errors for each column
        for i, column in enumerate(test_data.columns):
            # Get the forecast horizon for the current column
            forecast_horizon = len(forecasts[i].values)
            
            # Calculate forecast errors: actual values minus forecasted values
            errors_df[column] = test_data[column].values[-forecast_horizon:] - forecasts[i].values
        
        # Keep only the first `min_forecast_horizon` rows and reset index
        truncated_errors_df = errors_df.iloc[:min_forecast_horizon].reset_index(drop=True)
        
        # Append the DataFrame of errors for the current model to the list
        metrics_list.append(truncated_errors_df)
    stacked = pd.concat(metrics_list, axis=0)
    
    return stacked

def load_passengers_data() -> pd.DataFrame:
    return pd.read_csv('./data/passengers.csv', parse_dates=['Date'])
    
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
    weekly_store_category_sales = store_category_sales[store_category_sales.index > threshold_date].resample('W').sum()
    
    food_columns = [x for x in weekly_store_category_sales.columns if 'FOOD' in x]
    household_columns = [x for x in weekly_store_category_sales.columns if 'HOUSEHOLD' in x]
    hobbies_columns = [x for x in weekly_store_category_sales.columns if 'HOBBIES' in x]
    
    weekly_store_category_food_sales = weekly_store_category_sales[food_columns]
    weekly_store_category_household_sales = weekly_store_category_sales[household_columns]
    weekly_store_category_hobbies_sales = weekly_store_category_sales[hobbies_columns]
    
    return (
        weekly_store_category_food_sales,
        weekly_store_category_household_sales,
        weekly_store_category_hobbies_sales
        )


def normalized_weekly_store_category_household_sales() -> pd.DataFrame:
    
    _,weekly_store_category_household_sales,_ = load_m5_weekly_store_category_sales_data()
    
    df_temp = weekly_store_category_household_sales.copy().reset_index()
    df_temp['week'] = df_temp['date'].dt.strftime('%U').astype(int)
    df_temp['year'] = df_temp['date'].dt.strftime('%Y').astype(int)

    condition = (df_temp['year'] > df_temp['year'].min()) & (df_temp['year'] < df_temp['year'].max())
    df_temp_filtered = df_temp[condition]
    cols1 = [x for x in df_temp.columns if ('HOUSEHOLD' in x or 'year' in x) ]
    yearly_means = df_temp_filtered[cols1].groupby('year').mean()
    df_temp_filtered = df_temp_filtered.merge(yearly_means,  on='year', how = 'left', suffixes=('', '_yearly_mean'))
    for item in [x for x in df_temp.columns if 'HOUSEHOLD' in x ]:
        df_temp_filtered[item + '_normalized'] = df_temp_filtered[item] / df_temp_filtered[item + '_yearly_mean']

    return df_temp_filtered[[x for x in df_temp_filtered.columns if (('HOUSEHOLD' in x or 'week' in x or 'year' in x or 'date' in x) and ('yearly' not in x))]]



