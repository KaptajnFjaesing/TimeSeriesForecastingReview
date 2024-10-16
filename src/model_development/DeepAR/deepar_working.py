"""
Created on Tue Oct 15 19:30:26 2024

@author: Jonas Petersen
"""

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
from gluonts.time_feature.holiday import (
        squared_exponential_kernel,
        SpecialDateFeatureSet,
        NEW_YEARS_DAY,
        MARTIN_LUTHER_KING_DAY,
        SUPERBOWL,
        THANKSGIVING,
        CHRISTMAS_EVE,
        CHRISTMAS_DAY,
        NEW_YEARS_EVE
    )
from functools import lru_cache
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.npts import NPTSPredictor
from gluonts.evaluation import Evaluator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions.negative_binomial import NegativeBinomialOutput

import json

from src.data_loader import load_m5_data

@lru_cache(maxsize=None)
def get_sell_price(item_id, store_id, func=None):
    prices = calendar.merge(
        sell_prices.loc[item_id, store_id], on=["wm_yr_wk"], how="left"
    ).sell_price
    if (func=='diff'):
        return np.diff(prices)
    if (func=='rel'):
        return prices[:-1]/prices[0]
    return prices

sell_prices, sales_train_validation, calendar, submission_file = load_m5_data()
sales_train_evaluation = pd.read_csv('./data/sales_train_evaluation.csv')

calendar['event_type_1'] = calendar['event_type_1'].apply(lambda x: 0 if str(x)=="nan" else 1)
calendar['event_type_2'] = calendar['event_type_2'].apply(lambda x: 0 if str(x)=="nan" else 1)
calendar['snap_CA'] = calendar['snap_CA'].apply(lambda x: 0 if str(x)=="nan" else 1)
calendar['snap_TX'] = calendar['snap_TX'].apply(lambda x: 0 if str(x)=="nan" else 1)
calendar['snap_WI'] = calendar['snap_WI'].apply(lambda x: 0 if str(x)=="nan" else 1)


CUT = 1000

sales_train_validation  = sales_train_validation.set_index(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]).iloc[:CUT]
sales_train_validation.sort_index(inplace=True)

sales_train_evaluation  = sales_train_evaluation.set_index(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]).iloc[:CUT]
sales_train_evaluation.sort_index(inplace=True)

sell_prices  = sell_prices.set_index(["item_id", "store_id"])
sell_prices.sort_index(inplace=True)

sales_train_validation_hobbies1 = sales_train_validation[sales_train_validation.index.get_level_values('dept_id') =='HOBBIES_1'].copy()

sales_train_validation_hobbies1["state"] = pd.CategoricalIndex(sales_train_validation_hobbies1.index.get_level_values(5)).codes
sales_train_validation_hobbies1["store"] = pd.CategoricalIndex(sales_train_validation_hobbies1.index.get_level_values(4)).codes
sales_train_validation_hobbies1["cat"] = pd.CategoricalIndex(sales_train_validation_hobbies1.index.get_level_values(3)).codes
sales_train_validation_hobbies1["dept"] = pd.CategoricalIndex(sales_train_validation_hobbies1.index.get_level_values(2)).codes
sales_train_validation_hobbies1["item"] = pd.CategoricalIndex(sales_train_validation_hobbies1.index.get_level_values(1)).codes

sales_train_evaluation_hobbies1 = sales_train_evaluation[sales_train_evaluation.index.get_level_values('dept_id') =='HOBBIES_1'].copy()

sales_train_evaluation_hobbies1["state"] = pd.CategoricalIndex(sales_train_evaluation_hobbies1.index.get_level_values(5)).codes
sales_train_evaluation_hobbies1["store"] = pd.CategoricalIndex(sales_train_evaluation_hobbies1.index.get_level_values(4)).codes
sales_train_evaluation_hobbies1["cat"] = pd.CategoricalIndex(sales_train_evaluation_hobbies1.index.get_level_values(3)).codes
sales_train_evaluation_hobbies1["dept"] = pd.CategoricalIndex(sales_train_evaluation_hobbies1.index.get_level_values(2)).codes
sales_train_evaluation_hobbies1["item"] = pd.CategoricalIndex(sales_train_evaluation_hobbies1.index.get_level_values(1)).codes


#%% Add features
hdays = [
          NEW_YEARS_DAY,
          MARTIN_LUTHER_KING_DAY,
          SUPERBOWL,
          THANKSGIVING,
          CHRISTMAS_EVE,
          CHRISTMAS_DAY,
          NEW_YEARS_EVE]
kernel = squared_exponential_kernel(alpha=1.0)

sfs = SpecialDateFeatureSet(hdays, kernel)

time_index = pd.to_datetime(calendar.date)
date_indices = pd.date_range(
     start=time_index[0],
     periods=1969,
     freq='1D')

holiday_features = list(sfs(date_indices))


#%% prepare the categorical features

feat_static_cat = [
        {
            "name": "state_id",
            "cardinality": len(sales_train_validation_hobbies1["state"].unique()),
        },
        {
            "name": "store_id",
            "cardinality": len(sales_train_validation_hobbies1["store"].unique()),
        },
        {
            "name": "cat_id", "cardinality": len(sales_train_validation_hobbies1["cat"].unique())},
        {
            "name": "dept_id",
            "cardinality": len(sales_train_validation_hobbies1["dept"].unique()),
        },
        {
            "name": "item_id",
            "cardinality": len(sales_train_validation_hobbies1["item"].unique()),
        },
    ]

feat_dynamic_real = [
        {"name": "sell_price", "cardinality": 1},
        {"name": "event_1", "cardinality": 1},
        {"name": "event_2", "cardinality": 1},
        {"name": "snap", "cardinality": 1},
    ] + [
        {'name': hol, 'cardinality': 1} for hol in sfs.feature_names
    ]

stat_cat_cardinalities = [
                    len(sales_train_validation_hobbies1["state"].unique()), 
                    len(sales_train_validation_hobbies1["store"].unique()),
                    len(sales_train_validation_hobbies1["cat"].unique()),
                    len(sales_train_validation_hobbies1["dept"].unique()),
                    len(sales_train_validation_hobbies1["item"].unique())
                ]

#%% Prepare test and training data

date_columns = [x for x in sales_train_validation.columns if 'd_' in x]
dates = sales_train_validation[date_columns]
L = len(date_columns)

# Build training set
train_ds = []
train_ds = []
for index, item in tqdm(sales_train_validation_hobbies1.iterrows()):
    id, item_id, dept_id, cat_id, store_id, state_id = index
    start_index = np.nonzero(item[date_columns].values)[0][0]
    start_date = time_index[start_index]
    time_series = {}
    time_series["start"] = str(start_date)
    time_series["item_id"] = id[:-11]
    time_series["feat_static_cat"] = [
            item['state'],
            item['store'],
            item['cat'],
            item['dept'],
            item['item'],
        ]
    sell_price = get_sell_price(item_id, store_id)
    time_series["target"] = item.iloc[start_index:L].values.astype(np.float32).tolist()
    time_series["feat_dynamic_real"] = (
        np.concatenate(
                (
                    np.expand_dims(sell_price.iloc[start_index:L].values, 0),
                    np.expand_dims(calendar['event_type_1'].iloc[start_index:L].values, 0),
                    np.expand_dims(calendar['event_type_2'].iloc[start_index:L].values, 0),
                    np.expand_dims(calendar['snap_'+state_id].iloc[start_index:L].values, 0),
                    np.expand_dims(holiday_features[0][start_index:L], 0),
                    np.expand_dims(holiday_features[1][start_index:L], 0),
                    np.expand_dims(holiday_features[2][start_index:L], 0),
                    np.expand_dims(holiday_features[3][start_index:L], 0),
                    np.expand_dims(holiday_features[4][start_index:L], 0),
                    np.expand_dims(holiday_features[5][start_index:L], 0),
                    np.expand_dims(holiday_features[6][start_index:L], 0)

                ),
                0,
            )
        .astype(np.float32)
        .tolist()
    )

    train_ds.append(time_series.copy())

train_ds = ListDataset(train_ds, freq='1D')

date_columns_evaluation = [x for x in sales_train_evaluation_hobbies1.columns if 'd_' in x]
L2 = len(date_columns_evaluation)
# Build testing set
test_ds = []
for index, item in tqdm (sales_train_evaluation_hobbies1.iterrows()):
    id, item_id, dept_id, cat_id, store_id, state_id = index
    start_index = np.nonzero(item[date_columns_evaluation].values)[0][0]
    start_date = time_index[start_index]
    time_series = {}
    time_series["start"] = str(start_date)
    time_series["item_id"] = id[:-11]
    time_series["feat_static_cat"] = [
            item['state'],
            item['store'],
            item['cat'],
            item['dept'],
            item['item'],
        ]
    sell_price = get_sell_price(item_id, store_id)
    time_series["target"] = item.iloc[start_index:L2].values.astype(np.float32).tolist()
    time_series["feat_dynamic_real"] = (
        np.concatenate(
                (
                    np.expand_dims(sell_price.iloc[start_index:L2].values, 0),
                    np.expand_dims(calendar['event_type_1'].iloc[start_index:L2].values, 0),
                    np.expand_dims(calendar['event_type_2'].iloc[start_index:L2].values, 0),
                    np.expand_dims(calendar['snap_'+state_id].iloc[start_index:L2].values, 0),
                    np.expand_dims(holiday_features[0][start_index:L2], 0),
                    np.expand_dims(holiday_features[1][start_index:L2], 0),
                    np.expand_dims(holiday_features[2][start_index:L2], 0),
                    np.expand_dims(holiday_features[3][start_index:L2], 0),
                    np.expand_dims(holiday_features[4][start_index:L2], 0),
                    np.expand_dims(holiday_features[5][start_index:L2], 0),
                    np.expand_dims(holiday_features[6][start_index:L2], 0)

                ),
                0,
        )
        .astype(np.float32)
        .tolist()
    )

    test_ds.append(time_series.copy())

test_ds = ListDataset(test_ds, freq='1D')

#%% NPTS

prediction_length = 28

npts = NPTSPredictor(prediction_length=prediction_length)

# a GluonTS short-hand to make forecasts in a backtest scenario
forecast_npts_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=npts,
    num_samples=1000
)

# this can take a while to run
tss = list(tqdm(ts_it, total=len(test_ds)))
forecasts_npts = list(tqdm(forecast_npts_it, total=len(test_ds))) # list of n_time_series with elements of (number_of_samples,forecast_horizon)

# calculate metrics for NPTS
evaluator = Evaluator(quantiles = [0.1, 0.5, 0.9])
# gives me metrics both across time series and for individual time series
agg_metrics_npts, item_metrics_npts = evaluator(tss, forecasts_npts)

# %% deepAR

estimator_deepar = DeepAREstimator(
    prediction_length=prediction_length,
    freq="1D",
    distr_output=NegativeBinomialOutput(),
    trainer_kwargs={
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "max_epochs": 2,
        "logger": False,
        "callbacks": []
    },
    num_feat_dynamic_real = 11, 
    num_feat_static_cat = len(stat_cat_cardinalities),
    cardinality = stat_cat_cardinalities,
)

predictor_deepar = estimator_deepar.train(list(train_ds))

#%%
forecast_it, ts_it = make_evaluation_predictions(
    dataset = test_ds,
    predictor = predictor_deepar,
    num_samples = 1000
)

tss = list(tqdm(ts_it, total=len(test_ds)))
forecasts_deepar = list(tqdm(forecast_it, total=len(test_ds))) # list of n_time_series with elements of (number_of_samples,forecast_horizon)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
# gives me metrics both across time series and for individual time series
agg_metrics_deepar, item_metrics_deepar = evaluator(
    tss, forecasts_deepar)

#%%
print(json.dumps(agg_metrics_npts, indent=4))
print(json.dumps(agg_metrics_deepar, indent=4))


# %% Decision theory
from scipy.optimize import dual_annealing

def swish(x, b):
    # Compute the exponential in a numerically stable way
    # np.clip is used to limit the values of b*x to avoid overflow
    z = np.clip(b * x, -500, 500)  # Clip to prevent overflow in exp
    return (x / (1 + np.exp(-z))).round(3)


def cost_function(U, forecasts, H, t, k, psi, N0):
    b = 50
    Nt = N0 + U-np.cumsum(forecasts, axis=1)[:,np.newaxis,:]
    return k*np.dot(swish(Nt,b),H-t)+np.dot(swish(-Nt,b),psi)

def expected_cost(U, forecasts, H, t, k, psi, N0):
    return np.mean(cost_function(
        U = U,
        forecasts = forecasts,
        H = H,
        t = t,
        k = k,
        psi = psi,
        N0 = N0
    ), axis = 0)

#%%
H = 20

lead_time = 10
forecast_dates = forecasts_deepar[0].index.to_timestamp()[:lead_time]
forecasts = forecasts_deepar[0].samples[:,:lead_time]
true_timeseries_dates = tss[0].index.to_timestamp()
median_forecasts = np.median(forecasts, axis = 0)

training_data = tss[0].values[true_timeseries_dates<min(forecast_dates)]
test_data = tss[0].values[true_timeseries_dates>=min(forecast_dates)]

plt.figure()
plt.plot(true_timeseries_dates[true_timeseries_dates<min(forecast_dates)],training_data)
plt.plot(true_timeseries_dates[true_timeseries_dates>=min(forecast_dates)][:lead_time],test_data[:lead_time])
plt.plot(forecasts_deepar[0].index.to_timestamp()[:lead_time],median_forecasts)


#%%

t = np.arange(1,lead_time+1)
k = 5
psi = 1000*np.ones(len(t))
N0 = 1

possible_actions = np.arange(30)
U = np.zeros((len(possible_actions),lead_time))
U[:,-1] = possible_actions

b = 50
Nt = N0 + U-np.cumsum(forecasts, axis=1)[:,np.newaxis,:]

EC = expected_cost(U, forecasts, H, t, k, psi, N0)

# Create the plot
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(possible_actions, EC, color='blue', linewidth=2, label='Cost Function')  # Add line width and color
plt.xlabel('Possible Actions', fontsize=14)  # X-axis label
plt.ylabel('Cost', fontsize=14)  # Y-axis label

# Customize ticks and grid
plt.xticks(fontsize=12)  # X-axis ticks font size
plt.yticks(fontsize=12)  # Y-axis ticks font size
plt.grid(color='gray', linestyle='--', alpha=0.7)  # Add a grid for better readability

plt.xticks(possible_actions, fontsize=12)  # Set x-ticks from 0 to 10 with a step of 1
plt.yticks([])  # Clear y-ticks

# Add vertical lines
plt.axvline(U[np.argmin(EC),-1], color='red', linestyle='--', linewidth=2, label='Decision that minimize cost')  # Vertical line for minimum
plt.axvline(np.sum(median_forecasts), color='orange', linestyle='--', linewidth=2, label='Decision based on median')  # Another vertical line
plt.axvline(test_data[:lead_time].sum(), color='brown', linestyle='--', linewidth=2, label='Perfect decision in retrospective')  # Another vertical line


# Add a legend
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.figure()
plt.hist(np.cumsum(forecasts, axis=1)[:,-1])
plt.xlabel("Units removed from stock")



