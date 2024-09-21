# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:25:03 2024

@author: petersen.jonas
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import datetime
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sorcerer.sorcerer_model import SorcererModel

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO



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


def normalized_weekly_store_category_household_sales(df: pd.DataFrame) -> pd.DataFrame:
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
    return df_normalized


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


def generate_mean_profil_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        seasonality_period: int = 52,
        MA_window: int = 10
        ):
    
    model_forecasts_rolling = []
    model_forecasts_static = []
    value_columns = [x for x in df if x != 'date']
    df_temp = df.copy(deep=True)
    df_temp['week'] = df['date'].dt.strftime('%U').astype(int)
    df_temp['year'] = df['date'].dt.strftime('%Y').astype(int)
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        training_data = df_temp.iloc[:-fh]
        mean_profile, yearly_means = generate_mean_profile(
            df = training_data,
            seasonality_period = seasonality_period
            )
        #static forecast
        week_indices_in_forecast = [int(week - 1) for week in df_temp.iloc[-fh:]['week'].values]
        projected_scales_static = yearly_means[value_columns].diff().dropna().mean(axis = 0)+yearly_means[value_columns].iloc[-1]
        model_forecasts_static.append([pd.Series(row) for row in np.outer(projected_scales_static,mean_profile[week_indices_in_forecast])])

        # rolling forecast
        projected_scales_rolling = training_data[value_columns].iloc[-MA_window:].mean(axis = 0)
        model_forecasts_rolling.append([pd.Series(row) for row in np.outer(projected_scales_rolling,(mean_profile[week_indices_in_forecast]/mean_profile[week_indices_in_forecast][0]))])
        
    # Export data
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts_rolling,
             test_data = test_data[value_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle('./data/results/stacked_forecasts_rolling_mean.pkl')    
    
    compute_residuals(
             model_forecasts = model_forecasts_static,
             test_data = test_data[value_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle('./data/results/stacked_forecasts_static_mean.pkl')
    print("generate_mean_profil_stacked_forecast completed")


def generate_SSM_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        autoregressive: int = 2,
        seasonality_period: int = 52,
        harmonics: int = 10,
        level: bool = True,
        trend: bool = True,
        stochastic_level: bool = True, 
        stochastic_trend: bool = True,
        irregular: bool = True
        ):

    unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
    
    model_forecasts = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        training_data = df.iloc[:-fh].reset_index()
        y_train_min = training_data[unnormalized_column_group].min()
        y_train_max = training_data[unnormalized_column_group].max()
        y_train = (training_data[unnormalized_column_group]-y_train_min)/(y_train_max-y_train_min)
        model_denormalized = []
        for i in range(len(y_train.columns)):
            struobs = sm.tsa.UnobservedComponents(y_train[y_train.columns[i]],
                                                  level=level,
                                                  trend=trend,
                                                  autoregressive=autoregressive,
                                                  stochastic_level=stochastic_level,
                                                  stochastic_trend=stochastic_trend,
                                                  freq_seasonal=[{'period':seasonality_period,
                                                                  'harmonics':harmonics}],
                                                  irregular = irregular
                                                  )
            mlefit = struobs.fit(disp=0)
            model_results = mlefit.get_forecast(fh).summary_frame(alpha=0.05)
            model_denormalized.append(model_results['mean']*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]])
        model_forecasts.append(model_denormalized)
    
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
        
    compute_residuals(
             model_forecasts = model_forecasts,
             test_data = test_data[unnormalized_column_group],
             min_forecast_horizon = forecast_horizon
             ).to_pickle('./data/results/stacked_forecasts_statespace.pkl')
    print("generate_SSM_stacked_forecast completed")


def generate_abs_mean_gradient_training_data(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int
        ):
    unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
    df.iloc[:-(forecast_horizon+simulated_number_of_forecasts)].reset_index(drop = True)[unnormalized_column_group].diff().dropna().abs().mean(axis = 0).to_pickle('./data/results/abs_mean_gradient_training_data.pkl')
    print("generate_abs_mean_gradient_training_data completed")


def exponential_smoothing(
        df: pd.DataFrame,
        time_series_column: str,
        seasonality_period: int,
        forecast_periods: int
        ):
    df_temp = df.copy()
    df_temp.set_index('date', inplace=True)
    time_series = df_temp[time_series_column]
    model = ExponentialSmoothing(
        time_series, 
        seasonal='add',  # Use 'add' for additive seasonality or 'mul' for multiplicative seasonality
        trend='add',      # Use 'add' for additive trend or 'mul' for multiplicative trend
        seasonal_periods=seasonality_period,  # Adjust for the frequency of your seasonal period (e.g., 12 for monthly data)
        freq=time_series.index.inferred_freq
    )
    fit_model = model.fit()
    return fit_model.forecast(steps=forecast_periods)

def generate_exponential_smoothing_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        seasonality_period: int = 52,
        ):
    
    unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
    model_forecasts = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        training_data = df.iloc[:-fh].reset_index()
        y_train_min = training_data[unnormalized_column_group].min()
        y_train_max = training_data[unnormalized_column_group].max()
        
        y_train = (training_data[unnormalized_column_group]-y_train_min)/(y_train_max-y_train_min)
        y_train['date'] = training_data['date']
        model_denormalized = []
        time_series_columns = [x for x in y_train.columns if not 'date' in x]
        for i in range(len(time_series_columns)):
            model_results = exponential_smoothing(
                    df = y_train,
                    time_series_column = time_series_columns[i],
                    seasonality_period = seasonality_period,
                    forecast_periods = fh
                    )
            model_denormalized.append(model_results*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]])

        model_forecasts.append(model_denormalized)
    
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)

    compute_residuals(
             model_forecasts = model_forecasts,
             test_data = test_data[unnormalized_column_group],
             min_forecast_horizon = forecast_horizon
             ).to_pickle('./data/results/stacked_forecasts_exponential_smoothing.pkl')
    print("generate_exponential_smoothing_stacked_forecast completed")


# Wrapper to suppress output
def suppress_output(func, *args, **kwargs):
    f = StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        result = func(*args, **kwargs)
    return result

def generate_sorcerer_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        sampler_config: dict,
        model_config: dict
        ):

    time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
    unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
    df_time_series = df[time_series_columns]
    model_name = "SorcererModel"
    version = "v0.3"
    if sampler_config['sampler'] == "MAP":
        model_config['precision_target_distribution_prior_alpha'] = 1000
        model_config['precision_target_distribution_prior_beta'] = 0.01
    model_forecasts_sorcerer = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        training_data = df_time_series.iloc[:-fh]
        test_data = df_time_series.iloc[-fh:]
        y_train_min = training_data[unnormalized_column_group].min()
        y_train_max = training_data[unnormalized_column_group].max()
        sorcerer = SorcererModel(
            model_config = model_config,
            model_name = model_name,
            sampler_config = sampler_config,
            version = version
            )
        suppress_output(sorcerer.fit, training_data=training_data)
        (preds_out_of_sample, model_preds) = suppress_output(sorcerer.sample_posterior_predictive, test_data=test_data)
        model_forecasts_sorcerer.append([pd.Series((model_preds["target_distribution"].mean(("chain", "draw")).T)[i].values*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]]) for i in range(len(unnormalized_column_group))])
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts_sorcerer,
             test_data = test_data[unnormalized_column_group],
             min_forecast_horizon = forecast_horizon
             ).to_pickle(f"./data/results/stacked_forecasts_sorcerer_{sampler_config['sampler']}.pkl")
    print("generate_sorcerer_stacked_forecast completed")