"""
Created on Wed Sep 25 19:27:57 2024

@author: Jonas Petersen
"""

from src.data_loader import load_m5_weekly_store_category_sales_data
_, df, _ = load_m5_weekly_store_category_sales_data()

forecast_horizon = 26
simulated_number_of_forecasts = 50
number_of_weeks_in_a_year = 52.1429
context_length = 30
log_file = "./data/results/computation_times.json"
n_jobs = 10