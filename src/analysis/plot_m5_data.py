# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:23:03 2024

@author: petersen.jonas
"""

from src.data_loader import load_m5_weekly_store_category_sales_data
   
import matplotlib.pyplot as plt
import numpy as np

_,df,_ = load_m5_weekly_store_category_sales_data()

# %%

time_series_columns = [x for x in df.columns if 'HOUSEHOLD' in x]
n_cols = 2
n_rows = int(np.ceil(len(time_series_columns) / n_cols))
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 2 * n_rows), constrained_layout=True)
axs = axs.flatten()
for i in range(len(time_series_columns)):
    ax = axs[i]
    ax.set_title(time_series_columns[i])
    ax.plot(df['date'], df[time_series_columns[i]], color = 'tab:blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Sales')
    ax.grid(True)
plt.savefig('./figures/raw_data.png')
