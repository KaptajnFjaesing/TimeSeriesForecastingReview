import pandas as pd

def load_passengers_data() -> pd.DataFrame:
    return pd.read_csv('./data/passengers.csv', parse_dates=['Date'])
    

def load_sell_prices() -> pd.DataFrame:
    return pd.read_csv('./data/sell_prices.csv')