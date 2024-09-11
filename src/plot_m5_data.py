# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:23:03 2024

@author: petersen.jonas
"""

from src.load_data import load_m5_weekly_store_category_sales_data
   
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


_,weekly_store_category_household_sales,_ = load_m5_weekly_store_category_sales_data()


# Define the split date and discarded date
split_date = pd.Timestamp('2015-01-01')
discarded_date1 = pd.Timestamp('2011-12-31')
discarded_date2 = pd.Timestamp('2016-01-01')


# %%
# Plotting
plt.figure(figsize = (18,14))

for time_series_column in weekly_store_category_household_sales.columns:
    plt.plot(weekly_store_category_household_sales.index, weekly_store_category_household_sales[time_series_column], label = time_series_column)

# Set the title and labels
plt.title('Weekly Sales for each store-category', fontsize = 14)
plt.xlabel('Time', fontsize = 14)
plt.ylabel('Sales', fontsize = 14)

# Format x-axis to show only years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()

# Add vertical lines

plt.axvspan(weekly_store_category_household_sales.index.min(), discarded_date1, color='red', alpha=0.3, label='Discarded Data')
plt.axvspan(discarded_date2, weekly_store_category_household_sales.index.max(), color='red', alpha=0.3)
plt.axvline(x=split_date, color='black', linestyle='--', label='Train/Test Split')

plt.tick_params(axis='both', which='major', labelsize=14)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=10)  # Minor ticks


# Add legend
plt.legend(loc='upper left', fontsize = 14)

# Set grid
plt.grid(True)

plt.savefig('./docs/report/figures/raw_data.pdf')