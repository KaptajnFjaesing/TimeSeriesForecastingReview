# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:00:44 2024

@author: petersen.jonas
"""

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

from src.data_loader import load_m5_data

sell_prices, sales_train_validation, calendar, submission_file = load_m5_data()
sales_train_evaluation = pd.read_csv('./data/sales_train_evaluation.csv')

calendar['event_type_1'] = calendar['event_type_1'].apply(lambda x: 0 if str(x)=="nan" else 1)
calendar['event_type_2'] = calendar['event_type_2'].apply(lambda x: 0 if str(x)=="nan" else 1)
calendar['snap_CA'] = calendar['snap_CA'].apply(lambda x: 0 if str(x)=="nan" else 1)
calendar['snap_TX'] = calendar['snap_TX'].apply(lambda x: 0 if str(x)=="nan" else 1)
calendar['snap_WI'] = calendar['snap_WI'].apply(lambda x: 0 if str(x)=="nan" else 1)


CUT = 10000

sales_train_validation  = sales_train_validation.set_index(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]).iloc[:CUT]
sales_train_validation.sort_index(inplace=True)

sales_train_evaluation  = sales_train_evaluation.set_index(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]).iloc[:CUT]
sales_train_evaluation.sort_index(inplace=True)

sell_prices  = sell_prices.set_index(["item_id", "store_id"])
sell_prices.sort_index(inplace=True)


# %%

percentage_zeros = sales_train_validation[
    sales_train_validation == 0
    ].count(axis=1)/len(sales_train_validation.columns)
total_sales_by_item = sales_train_validation.sum(axis=1)

plt.figure()
plt.hist(percentage_zeros, bins = 100)

plt.figure()
plt.hist(total_sales_by_item, log = True, bins=np.logspace(0, 6, 100))
plt.xscale('log')

plt.figure()
plt.plot(sales_train_validation.sum(axis=0))
dates = pd.to_datetime(calendar.date.values[:1913])
years = dates.year
unique_years = pd.Series(years).drop_duplicates().index
year_labels = dates[unique_years].year
plt.xticks(unique_years, year_labels, rotation=45)


#%%

# we only look at FOODS_3 to make the data set a bit smaller for demo purposes
sales_train_validation = sales_train_validation[
    sales_train_validation.index.get_level_values(
        'dept_id') =='FOODS_3'
    ]
time_index = pd.to_datetime(calendar.date)

# Build static features
sales_train_validation["state"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(5)
    ).codes
sales_train_validation["store"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(4)
    ).codes
sales_train_validation["cat"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(3)
    ).codes
sales_train_validation["dept"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(2)
    ).codes
sales_train_validation["item"] = pd.CategoricalIndex(
        sales_train_validation.index.get_level_values(1)
    ).codes

sales_train_evaluation["state"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(5)
    ).codes
sales_train_evaluation["store"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(4)
    ).codes
sales_train_evaluation["cat"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(3)
    ).codes
sales_train_evaluation["dept"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(2)
    ).codes
sales_train_evaluation["item"] = pd.CategoricalIndex(
        sales_train_evaluation.index.get_level_values(1)
    ).codes

#%%

# Add features
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

date_indices = pd.date_range(
     start=time_index[0],
     periods=1969,
     freq='1D')

holiday_features = list(sfs(date_indices))

#%%

# let's look at a holiday indicator feature (ThanksGiving)
holiday_df = pd.DataFrame([date_indices, holiday_features[3]]).T
holiday_df.columns = ['date','holiday']
holiday_df.set_index('date', inplace=True) 

fig, ax = plt.subplots(3, 1, figsize=(15, 7))

holiday_df['2013-10-10':'2013-12-31'].plot(ax=ax[0])
holiday_df['2014-10-10':'2014-12-31'].plot(ax=ax[1])
holiday_df['2015-10-10':'2015-12-31'].plot(ax=ax[2])

plt.grid(which="both")
plt.show()



#%%

# prepare the categorical features
feat_static_cat = [
        {
            "name": "state_id",
            "cardinality": len(sales_train_validation["state"].unique()),
        },
        {
            "name": "store_id",
            "cardinality": len(sales_train_validation["store"].unique()),
        },
        {
            "name": "cat_id", "cardinality": len(sales_train_validation["cat"].unique())},
        {
            "name": "dept_id",
            "cardinality": len(sales_train_validation["dept"].unique()),
        },
        {
            "name": "item_id",
            "cardinality": len(sales_train_validation["item"].unique()),
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
                    len(sales_train_validation["state"].unique()), 
                    len(sales_train_validation["store"].unique()),
                    len(sales_train_validation["cat"].unique()),
                    len(sales_train_validation["dept"].unique()),
                    len(sales_train_validation["item"].unique())
                ]

#%%

from functools import lru_cache
from gluonts.dataset.common import ListDataset

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

# Build training set
train_ds = []
for index, item in tqdm(sales_train_validation.iterrows()):
    id, item_id, dept_id, cat_id, store_id, state_id = index
    start_index = np.nonzero(item.iloc[:1913].values)[0][0]
    start_date = time_index[start_index]
    time_series = {}

    state_enc, store_enc, cat_enc, dept_enc, item_enc = item.iloc[1913:]

    time_series["start"] = str(start_date)
    time_series["item_id"] = id[:-11]

    time_series["feat_static_cat"] = [
            state_enc,
            store_enc,
            cat_enc,
            dept_enc,
            item_enc,
        ]

    sell_price = get_sell_price(item_id, store_id)

    time_series["target"] = (
            item.iloc[start_index:1913].values.astype(np.float32).tolist()
    )
    time_series["feat_dynamic_real"] = (
        np.concatenate(
                (
                    np.expand_dims(sell_price.iloc[start_index:1913].values, 0),
                    np.expand_dims(calendar['event_type_1'].iloc[start_index:1913].values, 0),
                    np.expand_dims(calendar['event_type_2'].iloc[start_index:1913].values, 0),
                    np.expand_dims(calendar['snap_'+state_id].iloc[start_index:1913].values, 0),
                    np.expand_dims(holiday_features[0][start_index:1913], 0),
                    np.expand_dims(holiday_features[1][start_index:1913], 0),
                    np.expand_dims(holiday_features[2][start_index:1913], 0),
                    np.expand_dims(holiday_features[3][start_index:1913], 0),
                    np.expand_dims(holiday_features[4][start_index:1913], 0),
                    np.expand_dims(holiday_features[5][start_index:1913], 0),
                    np.expand_dims(holiday_features[6][start_index:1913], 0)

                ),
                0,
            )
        .astype(np.float32)
        .tolist()
    )

    train_ds.append(time_series.copy())
#%%
train_ds = ListDataset(train_ds, freq='1D')
#%%
from gluonts.dataset.common import load_datasets
from gluonts.dataset.field_names import FieldName
train_ds = ListDataset(train_ds, freq='1D')

#  plot price vs demand for a fast-moving item with price changes
elem = [s for s in train_ds if s['target'].sum() > 5000 and len(np.unique(s['feat_dynamic_real'][0])) > 5][0]

ex = pd.DataFrame([
    date_indices, 
    elem['target'], 
    elem['feat_dynamic_real'][0],
    elem['feat_dynamic_real'][1],
    elem['feat_dynamic_real'][3],
]).T
ex.columns = ['date','target','price','event','snap']
ex.set_index('date', inplace=True)

# Sales (blue) vs price changes (orange) vs events (black)
fig, ax = plt.subplots(2, 1, figsize=(7, 4))
ax2 = ax[0].twinx()
ax3 = ax[1].twinx()

ax[0].plot(ex['target'].loc['2012'], color='blue')
ax2.plot(ex['price'].loc['2012'], color='orange')

ax[1].plot(ex['event'].loc['2012'], color='black')

plt.grid(which="both")
plt.show()


#%%

# Build testing set
test_ds = []
for index, item in tqdm (sales_train_evaluation.iterrows()):
    id, item_id, dept_id, cat_id, store_id, state_id = index
    start_index = np.nonzero(item.iloc[:1941].values)[0][0]
    start_date = time_index[start_index]
    time_series = {}

    state_enc, store_enc, cat_enc, dept_enc, item_enc = item.iloc[1941:]

    time_series["start"] = str(start_date)
    time_series["item_id"] = id[:-11]

    time_series["feat_static_cat"] = [
            state_enc,
            store_enc,
            cat_enc,
            dept_enc,
            item_enc,
    ]

    sell_price = get_sell_price(item_id, store_id)

    time_series["target"] = (
        item.iloc[start_index:1941].values.astype(np.float32).tolist()
        )
    time_series["feat_dynamic_real"] = (
        np.concatenate(
                (
                    np.expand_dims(sell_price.iloc[start_index:1941].values, 0),
                    np.expand_dims(calendar['event_type_1'].iloc[start_index:1941].values, 0),
                    np.expand_dims(calendar['event_type_2'].iloc[start_index:1941].values, 0),
                    np.expand_dims(calendar['snap_'+state_id].iloc[start_index:1941].values, 0),
                    np.expand_dims(holiday_features[0][start_index:1941], 0),
                    np.expand_dims(holiday_features[1][start_index:1941], 0),
                    np.expand_dims(holiday_features[2][start_index:1941], 0),
                    np.expand_dims(holiday_features[3][start_index:1941], 0),
                    np.expand_dims(holiday_features[4][start_index:1941], 0),
                    np.expand_dims(holiday_features[5][start_index:1941], 0),
                    np.expand_dims(holiday_features[6][start_index:1941], 0)

                ),
                0,
        )
        .astype(np.float32)
        .tolist()
    )

    test_ds.append(time_series.copy())

test_ds = ListDataset(test_ds, freq='1D')

#%%


from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.npts import NPTSPredictor

prediction_length = 28

npts = NPTSPredictor(prediction_length=prediction_length)

# a GluonTS short-hand to make forecasts in a backtest scenario
forecast_npts_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=npts,
    num_samples=100
)

# this can take a while to run
tss = list(tqdm(ts_it, total=len(test_ds)))
forecasts_npts = list(tqdm(forecast_npts_it, total=len(test_ds)))


#%%

def plot_prob_forecasts2(ts_ent, forecast_ent):
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [
        f"{k}% prediction interval" for k in prediction_intervals
    ][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_ent[-plot_length:].plot(ax=ax, color='blue')  # plot the time series
    forecast_ent.plot(prediction_intervals=prediction_intervals, color='green')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


import matplotlib.pyplot as plt

def plot_prob_forecasts(ts_ent, forecast_ent):
    plot_length = 150
    prediction_intervals = (50.0, 70.0, 90.0)  # Adjust intervals as needed
    colors = ["green", "red", "brown"]  # Colors for different quantiles
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(ts_ent[-plot_length:].index.to_timestamp(), ts_ent[-plot_length:], color="blue", label="Observations")
    ax.plot(forecast_ent.index.to_timestamp(), forecast_ent.median, color="orange", label="Median Prediction")
    quantiles = [(100.0 - k) / 100 for k in prediction_intervals]
    for i, q in enumerate(quantiles):
        lower_quantile = forecast_ent.quantile(q)
        upper_quantile = forecast_ent.quantile(1 - q)
        ax.fill_between(
            forecast_ent.index.to_timestamp(),
            lower_quantile,
            upper_quantile,
            color=colors[i],  # Use different color for each quantile
            alpha=0.8,
            label=f"{prediction_intervals[i]}% Prediction Interval"
        )
    plt.grid(which="both")
    plt.legend(loc="upper left")
    plt.title("Time Series Forecast with Prediction Intervals")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()


ct = 0
to_plot = []
for idx in range(0, len(tss)):
    if (float(tss[idx].sum()) <= 1000.0):
        to_plot.append( [tss[idx], forecasts_npts[idx]] )
        ct += 1
    if ct > 10:
        break
    

#%%

plt_idx = 0
plot_prob_forecasts(to_plot[plt_idx][0], to_plot[plt_idx][1])

plt_idx = 1
plot_prob_forecasts(to_plot[plt_idx][0], to_plot[plt_idx][1])
# %%

# helper function to plot the aggregated values against aggregated forecasts
from gluonts.model.forecast import SampleForecast
# importing functools for reduce()
import functools

def sum_forecasts(left: SampleForecast, right: SampleForecast):
    return  SampleForecast(left.samples + right.samples,
        left.start_date,
        left.item_id,
        left.info
    )

# %%

agg_actuals = sum(tss)
agg_prob_forecasts = functools.reduce(lambda a, b: sum_forecasts(a, b), forecasts_npts)

# plot the aggregated sales against the aggregated forecast
plot_prob_forecasts(agg_actuals, agg_prob_forecasts)

# %%

# calculate metrics for NPTS
from gluonts.evaluation import Evaluator

evaluator = Evaluator(quantiles = [0.1, 0.5, 0.9])
# gives me metrics both across time series and for individual time series
agg_metrics_npts, item_metrics_npts = evaluator(tss, forecasts_npts)

# %%

import json
print(json.dumps(agg_metrics_npts, indent=4))

# %% deepAR

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions.negative_binomial import NegativeBinomialOutput

estimator_deepar = DeepAREstimator(
    prediction_length=prediction_length,
    freq="1D",
    # adjust the output
    distr_output=NegativeBinomialOutput(),
    trainer_kwargs={
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "max_epochs": 2,
    },
    num_feat_dynamic_real=11, 
    num_feat_static_cat=len(stat_cat_cardinalities),
    cardinality = stat_cat_cardinalities,
)
predictor_deepar = estimator_deepar.train(list(train_ds))


#%%

forecast_it, ts_it = make_evaluation_predictions(
    dataset = test_ds,
    predictor = predictor_deepar,
    num_samples = 100
)

#%%
print("Obtaining time series conditioning values ...")
tss = list(tqdm(ts_it, total=len(test_ds)))
#%%
print("Obtaining time series predictions ...")
forecasts_deepar = list(tqdm(forecast_it, total=len(test_ds)))

# %%

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
# gives me metrics both across time series and for individual time series
agg_metrics_deepar, item_metrics_deepar = evaluator(
    tss, forecasts_deepar)
agg_metrics_deepar

# %%

