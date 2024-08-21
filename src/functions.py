import pandas as pd

def load_passengers_data():
    df_passengers = pd.read_csv('../data/passengers.csv', parse_dates=['Date'])
    return df_passengers