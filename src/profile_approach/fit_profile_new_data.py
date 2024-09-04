# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:34:11 2024

@author: petersen.jonas
"""

from src.functions import (
    normalized_weekly_store_category_household_sales,
    load_m5_weekly_store_category_sales_data
    )
import matplotlib.pyplot as plt

df = normalized_weekly_store_category_household_sales()
_,weekly_store_category_household_sales,_ = load_m5_weekly_store_category_sales_data()


# %%
plt.figure()
plt.plot(df)
plt.title('Weekly Sales Normalized to yearly average')
plt.xlabel('Week')
plt.ylabel('Normalized Sales')
plt.grid(True)

plt.figure()
plt.plot(weekly_store_category_household_sales)
plt.title('Weekly Sales')
plt.xlabel('Week')
plt.ylabel('Sales')
plt.grid(True)
# %%

import numpy as np

n_weeks = 52

training_data = df.iloc[:-n_weeks]
test_data = df.iloc[-n_weeks:]

profile = [training_data[training_data.index.isocalendar().week == week].values.mean() for week in range(1,n_weeks+1)]

#%%
n_groups = len(df.columns)
ss_res_raw = np.ones((n_weeks,n_groups))
ss_tot_raw = np.ones((n_weeks,n_groups))
for week in range(1,n_weeks+1):
    
    df_test_week = test_data[test_data.index.isocalendar().week == week]
    if len(df_test_week) == 0:
        print("No test data")

    ss_res_raw[week-1] = ((df_test_week - profile[week-1])**2).values[0,:]
    ss_tot_raw[week-1] = ((df_test_week - df_test_week.mean(axis = 1).values[0])**2).values[0,:]

ss_res = ss_res_raw.mean(axis = 1)
ss_tot = ss_tot_raw.mean(axis = 1)

r_squared = 1-ss_res/ss_tot

print(r_squared.mean(), r_squared.std())


# %%

ko = test_data
ko['week_number'] = ko.index.isocalendar().week 
ko = ko.sort_values('week_number')

hest = ko[ko['week_number'] <= n_weeks][ko.columns[:-1]]


plt.figure()

# Line plot for 'profile'
plt.plot(range(1, n_weeks + 1), profile, label='Profile', linestyle='-')

# Scatter plot for 'hest'
for col in hest.columns[:-1]:  # Iterate over all columns except 'week_number'
    plt.scatter(range(1, n_weeks + 1), hest[col], alpha=0.1, color = 'b')

plt.xlabel('Week Number')
plt.ylabel('Values')
plt.title('Scatter Plot with Profile Line')
plt.legend()
plt.show()