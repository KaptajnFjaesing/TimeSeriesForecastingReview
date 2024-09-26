"""
Created on Wed Sep 25 12:01:39 2024

@author: Jonas Petersen
"""
from tqdm import tqdm
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import src.generate_stacked_residuals.global_model_parameters as gmp


model_config_default = {
    "max_depth": [7, 10],
    "num_leaves": [20, 50],
    "learning_rate": [0.01],
    "n_estimators": [100, 200],
    "colsample_bytree": [0.3, 0.5],
    'verbose':[-1]
}

def train_and_forecast(
        training_data,
        testing_data,
        column_name,
        model,
        cv_split,
        parameters
        ):
    X_train = training_data[["day_of_year", "month", "quarter", "year"]]
    y_train = training_data[column_name]
    X_test = testing_data[["day_of_year", "month", "quarter", "year"]]
    grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters)
    grid_search.fit(X_train, y_train)
    return grid_search.predict(X_test)


def generate_lgbm_w_sklearn_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        forecast_horizon: int = gmp.forecast_horizon,
        model_config: dict = model_config_default
        ):
    df_time_series = df.copy(deep = True)
    df_time_series["day_of_year"] = df_time_series["date"].dt.dayofyear
    df_time_series["month"] = df_time_series["date"].dt.month
    df_time_series["quarter"] = df_time_series["date"].dt.quarter
    df_time_series["year"] = df_time_series["date"].dt.year
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    residuals = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_lgbm_w_sklearn_stacked_residuals'):
        testing_data = df_time_series.iloc[-fh:].head(forecast_horizon)
        model_forecasts = pd.DataFrame({
            column: train_and_forecast(
                training_data = df_time_series.iloc[:-fh],
                testing_data  = testing_data,
                column_name = column,
                model = lgbm.LGBMRegressor(),
                cv_split = TimeSeriesSplit(n_splits=4, test_size=forecast_horizon),
                parameters = model_config
                ) for column in time_series_column_group
        })
        residuals.append((testing_data[time_series_column_group]-model_forecasts.set_index(df.iloc[-fh:].head(forecast_horizon).index)).reset_index(drop = True))
    pd.concat(residuals, axis=0).to_pickle('./data/results/stacked_residuals_lgbm_sklearn.pkl')

generate_lgbm_w_sklearn_stacked_residuals()