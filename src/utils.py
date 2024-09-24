"""
Created on Wed Sep 11 13:25:03 2024

@author: Jonas Petersen
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
import lightgbm as lgbm
import darts.models as dm
from darts import TimeSeries

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sorcerer.sorcerer_model import SorcererModel
from tlp_regression.tlp_regression_model import TlpRegressionModel

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
    household_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
    df.iloc[:-(forecast_horizon+simulated_number_of_forecasts)].reset_index(drop = True)[household_columns].diff().dropna().abs().mean(axis = 0).to_pickle('./data/results/abs_mean_gradient_training_data.pkl')
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
        seasonal_periods=int(seasonality_period),  # Adjust for the frequency of your seasonal period (e.g., 12 for monthly data)
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


# Sorcerer functions

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
    sorcerer = SorcererModel(
        model_config = model_config,
        model_name = "SorcererModel",
        model_version = "v0.3.1"
        )
    time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
    model_forecasts_sorcerer = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        training_data = df.iloc[:-fh]
        test_data = df.iloc[-fh:]
        y_train_min = training_data[time_series_columns].min()
        y_train_max = training_data[time_series_columns].max()
        suppress_output(func = sorcerer.fit, training_data=training_data, sampler_config = sampler_config)
        (preds_out_of_sample, model_preds) = suppress_output(sorcerer.sample_posterior_predictive, test_data=test_data)
        model_forecasts_sorcerer.append([pd.Series((model_preds["target_distribution"].mean(("chain", "draw")).T)[i].values*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]]) for i in range(len(time_series_columns))])
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts_sorcerer,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle(f"./data/results/stacked_forecasts_sorcerer_{sampler_config['sampler']}.pkl")
    print("generate_sorcerer_stacked_forecast completed")

# LIGHT GBM functions

def feature_generation(
        df: pd.DataFrame,
        time_series_column: str,
        context_length: int,
        seasonality_period: float):
    df_features = df.copy()
    df_features[time_series_column+'_log'] = np.log(df_features[time_series_column]) # Log-scale
    df_0 = df_features.iloc[0,:] # save first datapoint
    df_features[time_series_column+'_log_diff'] = df_features[time_series_column+'_log'].diff() # gradient
    for i in range(1,context_length):
        df_features[f'{i}'] = df_features[time_series_column+'_log_diff'].shift(i) # Fill lagged values as features    
    df_features.dropna(inplace=True)
    df_features['time_sine'] = np.sin(2*np.pi*df_features.index/seasonality_period) # Create a sine time-feature with the period set to 12 months
    
    return df_0, df_features

def light_gbm_predict(
        x_test,
        forecast_horizon,
        context_length,
        seasonality_period,
        model
        ):

    predictions = np.zeros(forecast_horizon)
    context = x_test.iloc[0].values.reshape(1,-1)
    test_start_index_plus_one = x_test.index[0]+1

    for i in range(forecast_horizon):
        predictions[i] = model.predict(context)[0]
        context[0,1:context_length] = context[0,0:context_length-1]
        context[0,0] = predictions[i]
        context[0,-1] = np.sin(2*np.pi*(test_start_index_plus_one+i)/seasonality_period) # this is for next iteration
    return predictions

def light_gbm_forecast(
        x_train:pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        time_series_column: str,
        forecast_horizon: int,
        context_length: int,
        seasonality_period: float,
        model_config: dict
        ):
    
    train_dataset = lgbm.Dataset(
        data = x_train.values,
        label = y_train.values
        )
    model = lgbm.train(
        params = model_config,
        train_set = train_dataset, 
        num_boost_round = 2000
        )

    predictions = light_gbm_predict(
        x_test = x_test,
        forecast_horizon = forecast_horizon,
        context_length = context_length,
        seasonality_period = seasonality_period,
        model = model
        )
    return predictions


def generate_light_gbm_stacked_forecast(
        df: pd.DataFrame,
        simulated_number_of_forecasts: int,
        forecast_horizon: int,
        context_length: int,
        seasonality_period: float,
        model_config: dict
        ):
    time_series_columns = [x for x in df.columns if not 'date' in x]
    feature_columns = [str(x) for x in range(1,context_length)]+['time_sine']
    model_forecasts = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        model_denormalized = []
        for i in range(len(time_series_columns)):
            df_0, df_features = feature_generation(
                df = df,
                time_series_column = time_series_columns[i],
                context_length = context_length,
                seasonality_period = seasonality_period
                )
            x_train = df_features[feature_columns].iloc[:-fh]
            y_train = df_features[time_series_columns[i]+'_log_diff'].iloc[:-fh]
            x_test = df_features[feature_columns].iloc[-fh:]
            baseline = df_features[time_series_columns[i]][x_test.index-1].iloc[0]
            model_results = light_gbm_forecast(
                    x_train = x_train,
                    y_train = y_train,
                    x_test = x_test,
                    time_series_column = time_series_columns[i],
                    forecast_horizon = forecast_horizon,
                    context_length = context_length,
                    seasonality_period = seasonality_period,
                    model_config = model_config
                    )
            model_denormalized.append(pd.Series(baseline*np.exp(np.cumsum(model_results))))
        model_forecasts.append(model_denormalized)
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle('./data/results/stacked_forecasts_light_gbm.pkl')
    print("generate_light_gbm_stacked_forecast completed")



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
    
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters)
    grid_search.fit(X_train, y_train)
    
    # Predict for the test set
    return grid_search.predict(X_test)

def generate_light_gbm_w_sklearn_stacked_forecast(
        df: pd.DataFrame,
        simulated_number_of_forecasts: int,
        forecast_horizon: int,
        model_config: dict
        ):
    time_series_columns = [x for x in df.columns if not 'date' in x]
    df_time_series = df.copy(deep = True)
    df_time_series["day_of_year"] = df_time_series["date"].dt.dayofyear
    df_time_series["month"] = df_time_series["date"].dt.month
    df_time_series["quarter"] = df_time_series["date"].dt.quarter
    df_time_series["year"] = df_time_series["date"].dt.year
    model_forecasts = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        results = pd.DataFrame({
            column: train_and_forecast(
                training_data = df_time_series.iloc[:-fh],
                testing_data  = df_time_series.iloc[-fh:],
                column_name = column,
                model = lgbm.LGBMRegressor(),
                cv_split = TimeSeriesSplit(n_splits=4, test_size=forecast_horizon),
                parameters = model_config
                ) for column in time_series_columns
        })
        model_forecasts.append(pd.Series([results[col] for col in time_series_columns]))
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle('./data/results/stacked_forecasts_light_gbm_w_sklearn.pkl')
    print("generate_light_gbm_w_sklearn_stacked_forecast completed")


def generate_tlp_regression_model_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        sampler_config: dict,
        model_config: dict,
        ):
    model_version = "v0.1.0"
    tlp_regression_model = TlpRegressionModel(
        model_config = model_config,
        model_name = "TLPRegressionModel",
        model_version = model_version
        )
    number_of_weeks_in_a_year = 52.1429
    number_of_months = 12
    number_of_quarters = 4
    time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
    df_time_series = df.copy(deep = True)
    df_time_series.loc[:, 'week'] = np.sin(2*np.pi*df_time_series['date'].dt.strftime('%U').astype(int)/number_of_weeks_in_a_year)
    df_time_series.loc[:, "month"] = np.sin(2*np.pi*df_time_series["date"].dt.month/number_of_months)
    df_time_series.loc[:, "quarter"] = np.sin(2*np.pi*df_time_series["date"].dt.quarter/number_of_quarters)
    model_forecasts_sorcerer = []
    for fh in tqdm(range(forecast_horizon, forecast_horizon + simulated_number_of_forecasts)):
        training_data = df_time_series.iloc[:-fh].copy(deep=True)
        test_data = df_time_series.iloc[-fh:].copy(deep=True)
        training_min_year = training_data["date"].dt.year.min()
        training_max_year = training_data["date"].dt.year.max()
        training_data.loc[:, "year"] = (training_data["date"].dt.year - training_min_year) / (training_max_year - training_min_year)
        test_data.loc[:, "year"] = (test_data["date"].dt.year - training_min_year) / (training_max_year - training_min_year)
        y_training_min = training_data[time_series_columns].min()
        y_training_max = training_data[time_series_columns].max()
        x_train = training_data[["week", "month", "quarter", "year"]].values
        y_train = ((training_data[time_series_columns]-y_training_min)/(y_training_max-y_training_min)).values
        x_test = test_data[["week", "month", "quarter", "year"]].values        
        suppress_output(tlp_regression_model.fit, X = x_train, y = y_train, sampler_config = sampler_config)
        model_preds = suppress_output(tlp_regression_model.sample_posterior_predictive, x_test=x_test)
        model_forecasts_sorcerer.append([pd.Series((model_preds["target_distribution"].mean(("chain", "draw")).T)[i].values*(y_training_max[y_training_max.index[i]]-y_training_min[y_training_min.index[i]])+y_training_min[y_training_min.index[i]]) for i in range(len(time_series_columns))])
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts_sorcerer,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle(f"./data/results/stacked_forecasts_tlp_regression_model_{sampler_config['sampler']}.pkl")
    print("tlp_regression_model completed")
    

def generate_naive_darts_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int
        ):
    naive_model = dm.NaiveDrift()
    time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
    model_forecasts = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        training_data = df.iloc[:-fh]
        test_data = df.iloc[-fh:]
        target = TimeSeries.from_dataframe(training_data, 'date')
        naive_model.fit(target)
        naive_predict = naive_model.predict(forecast_horizon)        
        model_forecasts.append([pd.Series(naive_predict.values()[:,i]) for i in range(len(time_series_columns))])
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle("./data/results/stacked_forecasts_naive_darts.pkl")
    print("generate_naive_darts_stacked_forecast completed")


def generate_lgbm_darts_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        model_config: dict
        ):
    time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
    model_forecasts = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        lgbm_model = dm.LightGBMModel(**model_config)
        lgbm_model.fit(TimeSeries.from_dataframe(df.iloc[:-fh], 'date'))
        lgbm_predict = lgbm_model.predict(forecast_horizon)    
        model_forecasts.append([pd.Series(lgbm_predict.values()[:,i]) for i in range(len(time_series_columns))])
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle("./data/results/stacked_forecasts_lgbm_darts.pkl")
    print("generate_lgbm_darts_stacked_forecast completed")


def generate_tide_darts_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        model_config: dict
        ):
    time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
    model_forecasts = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        tide_model = dm.TiDEModel(**model_config)
        tide_model.fit(TimeSeries.from_dataframe(df.iloc[:-fh], 'date'))
        tide_predict = tide_model.predict(forecast_horizon)    
        model_forecasts.append([pd.Series(tide_predict.values()[:,i]) for i in range(len(time_series_columns))])
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle("./data/results/stacked_forecasts_tide_darts.pkl")
    print("generate_tide_darts_stacked_forecast completed")

def generate_xgboost_darts_stacked_forecast(
        df: pd.DataFrame,
        forecast_horizon: int,
        simulated_number_of_forecasts: int,
        model_config: dict
        ):
    time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
    model_forecasts = []
    for fh in tqdm(range(forecast_horizon,forecast_horizon+simulated_number_of_forecasts)):
        xgbm_model = dm.XGBModel(**model_config)
        xgbm_model.fit(TimeSeries.from_dataframe(df.iloc[:-fh], 'date'))
        xgbm_predict = xgbm_model.predict(forecast_horizon)    
        model_forecasts.append([pd.Series(xgbm_predict.values()[:,i]) for i in range(len(time_series_columns))])
    test_data = df.iloc[-(forecast_horizon+simulated_number_of_forecasts):].reset_index(drop = True)
    compute_residuals(
             model_forecasts = model_forecasts,
             test_data = test_data[time_series_columns],
             min_forecast_horizon = forecast_horizon
             ).to_pickle("./data/results/stacked_forecasts_xgboost_darts.pkl")
    print("generate_xgboost_darts_stacked_forecast completed")