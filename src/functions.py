import pandas as pd

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


def load_m5_sales_data() -> pd.DataFrame:

    (sell_prices_df, train_sales_df, calendar_df, submission_file) = load_m5_data()


    d_cols = [c for c in train_sales_df.columns if 'd_' in c]

    # Group by 'state_id' and sum the sales across the specified columns in `d_cols`
    sales_sum_df = train_sales_df.groupby(['state_id'])[d_cols].mean().T


    # Merge the summed sales data with the calendar DataFrame on the 'd' column and set the index to 'date'
    sales_states_df = sales_sum_df.merge(calendar_df.set_index('d')['date'], 
                                   left_index=True, right_index=True, 
                                   validate="1:1").set_index('date')
    # Ensure that the index of your DataFrame is in datetime format
    sales_states_df.index = pd.to_datetime(sales_states_df.index)

    return sales_states_df

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


