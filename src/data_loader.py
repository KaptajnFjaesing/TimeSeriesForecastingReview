"""
Created on Tue Sep 24 19:01:11 2024

@author: Jonas Petersen
"""

import pandas as pd
import datetime
from typing import Tuple


def load_m5_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the M5 dataset files.

    Returns:
        Tuple containing:
        - sell_prices_df: DataFrame with sell prices data.
        - train_sales_df: DataFrame with sales training validation data.
        - calendar_df: DataFrame with calendar data.
        - submission_file: DataFrame with sample submission data.
    """
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


def load_m5_weekly_store_category_sales_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process and aggregate M5 sales data into weekly store-category sales.

    Returns:
        Tuple containing:
        - weekly_store_category_food_sales: DataFrame with weekly food sales.
        - weekly_store_category_household_sales: DataFrame with weekly household sales.
        - weekly_store_category_hobbies_sales: DataFrame with weekly hobbies sales.
    """
    (sell_prices_df, train_sales_df, calendar_df, submission_file) = load_m5_data()

    threshold_date = datetime.datetime(2011, 1, 1)

    d_cols = [c for c in train_sales_df.columns if 'd_' in c]

    train_sales_df['store_cat'] = train_sales_df['store_id'].astype(str) + '_' + train_sales_df['cat_id'].astype(str)
    # Group by 'store_cat' and sum the sales across the specified columns in `d_cols`
    sales_sum_df = train_sales_df.groupby(['store_cat'])[d_cols].sum().T

    # Merge the summed sales data with the calendar DataFrame on the 'd' column and set the index to 'date'
    store_category_sales = sales_sum_df.merge(
        calendar_df.set_index('d')['date'],
        left_index=True,
        right_index=True,
        validate="1:1"
    ).set_index('date')

    # Ensure that the index of your DataFrame is in datetime format
    store_category_sales.index = pd.to_datetime(store_category_sales.index)
    weekly_store_category_sales = store_category_sales[
        store_category_sales.index > threshold_date
    ].resample('W-MON', closed="left", label="left").sum().iloc[1:]

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
