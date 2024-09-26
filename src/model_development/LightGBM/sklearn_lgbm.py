# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 08:44:01 2024

@author: petersen.jonas
"""

import numpy as np
import pandas as pd
from src.utils import load_m5_weekly_store_category_sales_data
import lightgbm as lgbm
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def train_and_forecast(training_data,column_name, model, cv_split, parameters):
    X_train = training_data[["day_of_year", "month", "quarter", "year"]]
    y_train = training_data[column_name]
    
    X_test = testing_data[["day_of_year", "month", "quarter", "year"]]
    
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters)
    grid_search.fit(X_train, y_train)
    
    # Predict for the test set
    return grid_search.predict(X_test)

_,df,_ = load_m5_weekly_store_category_sales_data()

time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

# %%
df_time_series = df[time_series_columns]
df_time_series["day_of_year"] = df_time_series["date"].dt.dayofyear
df_time_series["month"] = df_time_series["date"].dt.month
df_time_series["quarter"] = df_time_series["date"].dt.quarter
df_time_series["year"] = df_time_series["date"].dt.year

training_data = df_time_series.iloc[:-forecast_horizon]
testing_data = df_time_series.iloc[-forecast_horizon:]

cv_split = TimeSeriesSplit(n_splits=4, test_size=10)
model = lgb.LGBMRegressor()
parameters = {
    "max_depth": [4, 7, 10],
    "num_leaves": [20, 50],
    "learning_rate": [0.01, 0.05],
    "n_estimators": [100, 400],
    "colsample_bytree": [0.3, 0.5]
}

results = pd.DataFrame({
    column: train_and_forecast(training_data = training_data,column_name = column, model = model, cv_split = cv_split, parameters = parameters) for column in unnormalized_column_group
})


# %%
import matplotlib.pyplot as plt
import numpy as np

# Assuming these variables are defined:
# - X_train: Training data features
# - X_test: Testing data features
# - y_train: Training data targets
# - y_test: Testing data targets
# - renormalized_predictions: Model's forecasts


# Calculate the number of rows needed for 2 columns
n_cols = 2
n_rows = int(np.ceil(len(unnormalized_column_group) / n_cols))

# Create subplots with 2 columns and computed rows   
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Plot each time series
for i in range(len(unnormalized_column_group)):
    ax = axs[i]  # Get the correct subplot
    column_name = unnormalized_column_group[i]
    
    # Plot the training data
    ax.plot(training_data['date'], training_data[column_name], color='tab:red', label='Training Data')
    
    # Plot the test data
    ax.plot(testing_data['date'], testing_data[column_name], color='black', label='Test Data')
    
    # Plot the predictions (renormalized forecasts)
    ax.plot(testing_data['date'], results[column_name], color='tab:blue', label='Forecast')

    # Set title, labels, and grid
    ax.set_title(column_name)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes

# Show the final plot
plt.show()


# %%
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error
    )

def evaluate_model(y_test, prediction):
  print(f"MAE: {mean_absolute_error(y_test, prediction)}")
  print(f"MSE: {mean_squared_error(y_test, prediction)}")
  print(f"MAPE: {mean_absolute_percentage_error(y_test, prediction)}")

def plot_predictions(testing_dates, y_test, prediction):
  df_test = pd.DataFrame({"date": testing_dates, "actual": y_test, "prediction": prediction })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_test.plot(ax=ax, label="Actual", x="date", y="actual")
  df_test.plot(ax=ax, label="Prediction", x="date", y="prediction")
  plt.legend(["Actual", "Prediction"])
  plt.show()

plot_predictions(testing_dates = testing_data["date"], y_test = y_test, prediction = prediction)
evaluate_model(y_test = y_test, prediction = prediction)
# %%

seasonality_period = 52
min_forecast_horizon = 26
max_forecast_horizon = 52
context_length = 50


forecast_horizon = 26

training_data = df_time_series.iloc[:-forecast_horizon]

data_min = training_data.min()
data_max = training_data.max()


i = 0
(x_train, y_train) = create_lagged_features(
    df = training_data,
    time_series_columns = [unnormalized_column_group[i]],
    context_length = context_length,
    forecast_horizon = forecast_horizon,
    seasonality_period = seasonality_period,
    data_min = data_min,
    data_max = data_max
    )

(x_total,y_total) = create_lagged_features(
     df = df_time_series,
     time_series_columns = [unnormalized_column_group[i]],
     context_length = context_length,
     forecast_horizon = forecast_horizon,
     seasonality_period = seasonality_period,
     data_min = data_min,
     data_max = data_max
    )

# Split into train and validation set for early stopping
split_index = int(0.8 * len(x_train))
x_train_train, x_train_valid = x_train[:split_index], x_train[split_index:]
y_train_train, y_train_valid = y_train.iloc[:split_index], y_train.iloc[split_index:]

train_dataset = lgbm.Dataset(x_train_train, label=y_train_train[step].values.flatten(), free_raw_data=False)
valid_dataset = lgbm.Dataset(x_train_valid, label=y_train_valid[step].values.flatten(), free_raw_data=False)


x_test = x_total[x_total.index == training_data.iloc[-1:].index[0]]
# %%

params = {
    'verbose': -1,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 100,
    'learning_rate': 0.1,
    'feature_fraction': 0.8
}

models = []
predictions = []
renormalized_predictions = []
for step in y_train.columns:

    train_dataset = lgbm.Dataset(
        data = x_train.values,
        label = y_train[step].values.flatten()
        )
    
    model = lgbm.train(
        params,
        train_dataset,
        num_boost_round=1000,  # Increase num_boost_round for potential improvements
        valid_sets=[train_dataset, valid_dataset],
        valid_names=['train', 'valid']
        )
    
    prediction = model.predict(x_test.values)
    
    renormalized_prediction = prediction*(data_max[unnormalized_column_group[i]]-data_min[unnormalized_column_group[i]])+data_min[unnormalized_column_group[i]]
    
    predictions.append(prediction)
    renormalized_predictions.append(renormalized_prediction)
    

# %%

normalized_ts = (df_time_series[unnormalized_column_group[i]]-data_min[unnormalized_column_group[i]])/(data_max[unnormalized_column_group[i]]-data_min[unnormalized_column_group[i]])
c1 = training_data.iloc[-1:].index[0]
c2 = c1+forecast_horizon


plt.figure()
plt.plot(np.arange(x_test.index[0]-context_length,x_test.index[0]+1),x_test[x_test.columns[:-1]].values[0])
plt.plot(normalized_ts[(normalized_ts.index >= c1)].index[:-1],predictions)
plt.plot(normalized_ts, alpha = 0.4)

# %%

plt.figure()



plt.plot(y_total[(y_total.index < c1)].index, train_predict)
plt.plot(y_total[(y_total.index >= c1)].index, predictions)
plt.plot(y_total)

#%%
test_data = df_time_series.iloc[-forecast_horizon:]

import matplotlib.pyplot as plt

plt.figure()
plt.plot(renormalized_predictions)
plt.plot(test_data[unnormalized_column_group[i]].values)



# %%


seasonality_period = 52
min_forecast_horizon = 26
max_forecast_horizon = 52
context_length = 30

feature_columns = [str(x) for x in range(1,context_length)]+['time_sine']

model_forecasts_lgbm = []
for forecast_horizon in tqdm(range(min_forecast_horizon,max_forecast_horizon+1)):
    training_data = df_time_series.iloc[:-forecast_horizon]
    test_data = df_time_series.iloc[-forecast_horizon:]
    
    rescaled_forecasts = np.ones((len(unnormalized_column_group),forecast_horizon))
    model_denormalized = []
    for i in range(len(unnormalized_column_group)):
        
        
        (x_train, y_train) = create_lagged_features(
            df = df_time_series,
            time_series_columns = unnormalized_column_group[i],
            context_length = context_length,
            forecast_horizon = forecast_horizon,
            seasonality_period = seasonality_period
            )
        
        
        (x_total,y_total) = create_lagged_features(
             df = df_time_series,
             time_series_columns = [time_series],
             context_length = context_length,
             forecast_horizon = forecast_horizon,
             seasonality_period = seasonality_period
            )
        
        
        
         prognosticator.fit(
             X = x_train,
             y = y_train
             )
         model_preds = prognosticator.sample_posterior_predictive(x_test = x_total[x_total.index == training_data.iloc[-1:].index[0]])
        
        # Create LightGBM datasets
        train_dataset = lgbm.Dataset(
            data = x_train.values,
            label = y_train.values
            )
            
        # Define model parameters
        params = {
            'verbose': -1,
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 5,
            'learning_rate': 0.1,
            'feature_fraction': 0.9
        }
            
        # Train the model with early stopping
        model = lgbm.train(
            params,
            train_dataset, 
            num_boost_round = 2000
            )
        
        train_predict = model.predict(x_train.values)
        
        predictions = light_gbm_predict(
            x_test = x_test,
            forecast_horizon = forecast_horizon,
            context_length = context_length,
            seasonality_period = seasonality_period,
            model = model
            )
        
        baseline = df_features[unnormalized_column_group[i]][x_test.index-1].iloc[0]    
        model_denormalized.append(pd.Series(baseline*np.exp(np.cumsum(predictions))))
    model_forecasts_lgbm.append(model_denormalized)