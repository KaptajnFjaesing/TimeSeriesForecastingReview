import pandas as pd
import datetime

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
    
    # The first datapoint is not included, because it may be a part of a week only
    weekly_store_category_food_sales = weekly_store_category_sales[food_columns].iloc[1:]
    weekly_store_category_household_sales = weekly_store_category_sales[household_columns].iloc[1:]
    weekly_store_category_hobbies_sales = weekly_store_category_sales[hobbies_columns].iloc[1:]
    
    return (
        weekly_store_category_food_sales,
        weekly_store_category_household_sales,
        weekly_store_category_hobbies_sales
        )


def normalized_weekly_store_category_household_sales() -> pd.DataFrame:
    
    _,weekly_store_category_household_sales,_ = load_m5_weekly_store_category_sales_data()
    weekly_store_category_household_sales = weekly_store_category_household_sales[weekly_store_category_household_sales.index < datetime.datetime(2016, 1, 1)]
    df_temp = weekly_store_category_household_sales.copy()
    df_temp['year'] = df_temp.index.isocalendar().year
    yearly_means = df_temp.groupby('year').mean()
    df_temp = df_temp.reset_index().merge(yearly_means,  on='year', how = 'left', suffixes=('', '_yearly_mean')).set_index('date')

    for item in weekly_store_category_household_sales.columns:
        df_temp[item + '_normalized'] = df_temp[item] / df_temp[item + '_yearly_mean']

    return df_temp[[x for x in df_temp.columns if '_normalized' in x]]
    
def feature_engineering(
        df: pd.DataFrame(),
        column_list: list,
        context_length: int,
        train_length: int,
        period: int
        ):
    
    df_features = pd.DataFrame(index = df.index[:train_length])
    for column in column_list:
        gradient_column = df[column].diff()
        gradient_column.name = column+'_grad'
        df_features = pd.concat([df_features, gradient_column], axis=1)
        for i in range(1,context_length):
            shifted_col = df[column].diff().shift(i)
            shifted_col.name = column + f'_{i}'  # Name the new column
            df_features = pd.concat([df_features, shifted_col], axis=1)

    df_features.dropna(inplace=True)
    return df_features, df[column_list].iloc[0].values


