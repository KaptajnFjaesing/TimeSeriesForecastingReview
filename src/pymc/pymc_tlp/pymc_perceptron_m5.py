# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:58:48 2024

@author: petersen.jonas
"""

from src.utils import normalized_weekly_store_category_household_sales

df = normalized_weekly_store_category_household_sales()

import numpy as np
from tqdm import tqdm
import pandas as pd

seasonality_period = 52
min_forecast_horizon = 26
max_forecast_horizon = 52
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
normalized_column_group = [x for x in df.columns if '_normalized' in x ]

model_forecasts = []

for forecast_horizon in tqdm(range(min_forecast_horizon,52+1)):

    training_data = df.iloc[:-forecast_horizon].reset_index()
    y_train_min = training_data[unnormalized_column_group].min()
    y_train_max = training_data[unnormalized_column_group].max()
    
    y_train = (training_data[unnormalized_column_group]-y_train_min)/(y_train_max-y_train_min)
    y_train['date'] = training_data['date']
    model_denormalized = []
    
    time_series_columns = [x for x in y_train.columns if not 'date' in x]
    for i in range(len(time_series_columns)):
        
        # TLP model
        model_results = exponential_smoothing(
                df = y_train,
                time_series_column = time_series_columns[i],
                seasonality_period = seasonality_period,
                forecast_periods = forecast_horizon
                )
        model_denormalized.append(model_results*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]])

    model_forecasts.append(model_denormalized)