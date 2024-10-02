"""
Created on Wed Sep 25 10:25:25 2024

@author: Jonas Petersen
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.generate_stacked_residuals.global_model_parameters as gmp

def generate_mean_profile(df, seasonality_period): 
    df_temp = df.copy(deep=True)
    df_temp['week'] = df['date'].dt.strftime('%U').astype(int)
    df_temp['year'] = df['date'].dt.strftime('%Y').astype(int)

    df_temp = df_temp[df_temp['week'] != 53]
    weeks_per_year = df_temp.groupby('year').week.nunique()
    years_with_52_weeks = weeks_per_year[weeks_per_year == 52].index
    df_full_years = df_temp[df_temp['year'].isin(years_with_52_weeks)]
    yearly_means = df_full_years[[x for x in df_full_years.columns if ('HOUSEHOLD' in x or 'year' in x) ]].groupby('year').mean().reset_index()
    df_full_years_merged = df_full_years.merge(yearly_means,  on='year', how = 'left', suffixes=('', '_yearly_mean'))
    for item in [x for x in df_temp.columns if 'HOUSEHOLD' in x ]:
        df_full_years_merged[item + '_normalized'] = df_full_years_merged[item] / df_full_years_merged[item + '_yearly_mean']
    df_normalized = df_full_years_merged[[x for x in df_full_years_merged.columns if ('normalized' in x or x == 'week' or x == 'year')]]
    normalized_column_group = [x for x in df_normalized.columns if 'normalized' in x]
    df_melted = df_normalized.melt(
        id_vars='week',
        value_vars= normalized_column_group,
        var_name='Variable',
        value_name='Value'
        )
    seasonality_profile = np.array([df_melted['Value'][df_melted['week'] == week].values.mean() for week in range(1,int(seasonality_period)+1)])
    return seasonality_profile, yearly_means

def generate_mean_profile_stacked_residuals(
        df: pd.DataFrame = gmp.df,
        forecast_horizon: int = gmp.forecast_horizon,
        simulated_number_of_forecasts: int = gmp.simulated_number_of_forecasts,
        seasonality_period: int = int(gmp.number_of_weeks_in_a_year),
        MA_window: int = 10
        ):
    residuals_static = []
    residuals_rolling = []
    time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]
    df_temp = df.copy(deep=True)
    df_temp['week'] = df['date'].dt.strftime('%U').astype(int)
    df_temp['year'] = df['date'].dt.strftime('%Y').astype(int)
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts), desc = 'generate_mean_profile_stacked_residuals'):
        training_data = df_temp.iloc[:-fh]
        mean_profile, yearly_means = generate_mean_profile(
            df = training_data,
            seasonality_period = seasonality_period
            )
        #static forecast
        week_indices_in_forecast = [int(week - 1) for week in df_temp.iloc[-fh:].head(forecast_horizon)['week'].values]
        projected_scales_static = yearly_means[time_series_column_group].diff().dropna().mean(axis = 0)+yearly_means[time_series_column_group].iloc[-1]
        
        model_forecasts_static = pd.DataFrame(
            data = np.outer(
                mean_profile[week_indices_in_forecast],
                projected_scales_static
                ),
            columns = projected_scales_static.index)

        # rolling forecast
        projected_scales_rolling = training_data[time_series_column_group].iloc[-MA_window:].mean(axis = 0)
        """
            NOTE: 
            week_indices_in_forecast[0] gives the first week in the forecast, we need to subtract 1 to get the last week in the training data
            giving us week_indices_in_forecast[0] - 1. The logic for selecting the week in the training to use as the average is then calculated as
            -(np.ceil(MA_window/2) - 1), where -1 is used to included the last week in the training data.
            Note that this expression could be reduced, but is not to keep the code more understandable.
        """
        model_forecasts_rolling = pd.DataFrame(
            data = np.outer(
                (mean_profile[week_indices_in_forecast]/mean_profile[week_indices_in_forecast[0] - 1 - (int(np.ceil(MA_window/2)) - 1)]),
                projected_scales_rolling
                ),
            columns = projected_scales_rolling.index)
        
        test_data = df_temp[[x for x in df_temp.columns if 'HOUSEHOLD' in x]].iloc[-fh:].head(forecast_horizon).reset_index(drop = True)
        residuals_static.append((test_data-model_forecasts_static).reset_index(drop = True))
        residuals_rolling.append((test_data-model_forecasts_rolling).reset_index(drop = True))

    pd.concat(residuals_static, axis=0).to_pickle('./data/results/stacked_residuals_static_mean.pkl')
    pd.concat(residuals_rolling, axis=0).to_pickle('./data/results/stacked_residuals_rolling_mean.pkl')

generate_mean_profile_stacked_residuals()