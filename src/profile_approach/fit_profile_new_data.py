# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:34:11 2024

@author: petersen.jonas
"""

from src.functions import load_m5_sales_data
import matplotlib.pyplot as plt
import datetime

sales_states_df = load_m5_sales_data()

#%%
threshold_date = datetime.datetime(2011, 2, 1)

# Filter the DataFrame to keep only rows with dates greater than the threshold
df = sales_states_df[sales_states_df.index > threshold_date]
df_weekly_sales = df.resample('W').sum()

n_weeks = 52
states = df_weekly_sales.columns
training_data = df_weekly_sales.iloc[:-n_weeks-1]
test_data = df_weekly_sales.iloc[-n_weeks-1:]


plt.figure()
plt.plot(df_weekly_sales)
plt.title('Weekly Sales')
plt.xlabel('Week')
plt.ylabel('Sales')
plt.grid(True)
# %%




import numpy as np

profile = [training_data[training_data.index.isocalendar().week == week].values.mean() for week in range(1,n_weeks+1)]

#%%
n_states = len(states)
ss_res_raw = np.ones((n_weeks,n_states))
ss_tot_raw = np.ones((n_weeks,n_states))
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
plt.plot(range(1,n_weeks+1),profile)
plt.plot(range(1,n_weeks+1),hest)