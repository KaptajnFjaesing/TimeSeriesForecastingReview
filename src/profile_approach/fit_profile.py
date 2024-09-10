# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:34:11 2024

@author: petersen.jonas
"""

from src.load_data import (
    normalized_weekly_store_category_household_sales,
    load_m5_weekly_store_category_sales_data
    )
import matplotlib.pyplot as plt

df = normalized_weekly_store_category_household_sales()
_,weekly_store_category_household_sales,_ = load_m5_weekly_store_category_sales_data()


# %%
import numpy as np


n_weeks = 52
normalized_column_group = [x for x in df.columns if '_normalized' in x ]

training_data = df.iloc[:-n_weeks]
test_data = df.iloc[-n_weeks:]

future_mean_model = test_data[normalized_column_group].mean(axis = 1)

df_melted = training_data.melt(
    id_vars='week',
    value_vars= normalized_column_group,
    var_name='Variable',
    value_name='Value'
    )


#%% Mean profile

mean_profile = [df_melted['Value'][df_melted['week'] == week].values.mean() for week in range(1,n_weeks+1)]


#%% Fourier profile

def fourier_series(t, n_terms, T):
    terms = [np.ones(t.shape)]  # a_0 term
    for k in range(1, n_terms + 1):
        terms.append(np.cos(2 * np.pi * k * t / T))
        terms.append(np.sin(2 * np.pi * k * t / T))
    return np.column_stack(terms)

x_train = df_melted['week'].values
y_train = df_melted['Value'].values

# Number of terms in the Fourier series
fourier_terms = 10  # Adjust the number of terms for fitting
# Construct the design matrix for the Fourier series using the original data
# It has dimensions (data, coefficients) where the latter is 1+2*n_terms
data_x_training = fourier_series(
    t = x_train,
    n_terms = fourier_terms,
    T = n_weeks 
    )

# Perform the linear regression manually (least squares solution)
beta = np.linalg.inv(data_x_training.T @ data_x_training) @ data_x_training.T @ y_train

week_range = np.linspace(1, n_weeks+1, n_weeks)  # Go slightly beyond 12 to ensure smooth wrapping

week_range_data = fourier_series(
    t = week_range,
    n_terms = fourier_terms,
    T = n_weeks 
    )

fourier_profile = week_range_data @ beta



# %% Calculate r-squared

def test_r_squared_profile(
        test_data,
        profile
        ):
    n_weeks = len(profile)
    normalized_column_group = [x for x in test_data.columns if '_normalized' in x ]
    
    n_groups = len(normalized_column_group)
    ss_res_raw = np.ones((n_weeks,n_groups))
    ss_tot_raw = np.ones((n_weeks,n_groups))
    future_mean_model = test_data[normalized_column_group].mean(axis = 1)
    for week in range(1,n_weeks+1):
        df_test_week = test_data[test_data['week'] == week][normalized_column_group]
        if len(df_test_week) == 0:
            print("No test data")
            return np.nan, np.nan
        if len(profile) == 0:
            print("No profile data")
            return np.nan, np.nan
        ss_res_raw[week-1] = ((df_test_week - profile[week-1])**2).values[0,:]
        ss_tot_raw[week-1] = ((df_test_week - future_mean_model.iloc[week-1])**2).values[0,:]
        
    ss_res = ss_res_raw.mean(axis = 1)
    ss_tot = ss_tot_raw.mean(axis = 1)

    return ss_res, ss_tot

mean_profile_r_squared = test_r_squared_profile(
        test_data = test_data,
        profile = mean_profile
        )

fourier_profile_r_squared = test_r_squared_profile(
        test_data = test_data,
        profile = fourier_profile
        )


mean_r = 1-mean_profile_r_squared[0]/mean_profile_r_squared[1]
fourier_r = 1-fourier_profile_r_squared[0]/fourier_profile_r_squared[1]

print("mean r-squared mean model: ",mean_r.mean().round(2)," +- ",mean_r.std().round(2))
print("mean r-squared Fourier model: ",fourier_r.mean().round(2)," +- ",fourier_r.std().round(2))


# %% Plot profiles against future mean and test data as sanity check


plt.figure(figsize = (15,8))

# Line plot for 'profile'
plt.plot(test_data['date'], mean_profile, label='Mean model', linestyle='-')
plt.plot(test_data['date'], fourier_profile, label='Fourier Model', linestyle='-')
plt.plot(test_data['date'], future_mean_model, label='Future mean model', linestyle='-')

#plt.plot(test_data['date'], fourier_profile_r_squared[2]*0.01+1, label='Fourier model r_squared', linestyle='-')
#plt.plot(test_data['date'], mean_profile_r_squared[2]*0.01+1, label='mean model r_squared', linestyle='-')

# Scatter plot for 'hest'
for col in normalized_column_group:  # Iterate over all columns except 'week_number'
    plt.scatter(test_data['date'], test_data[col], alpha=0.1, color = 'b')

plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Scatter Plot with Profile Line')


# Add legend
plt.legend(loc='upper left')

# Set grid
plt.grid(True)

plt.savefig('./docs/report/figures/profile_plot.pdf')

# %% The scale for total sales, corresponding to the expected sales across the year

unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]


projected_scales = []
for col in unnormalized_column_group:

    yearly_averages = training_data[['year']+[col]].groupby('year').mean()
    
    mean_grad = training_data[['year']+[col]].groupby('year').mean().diff().dropna().mean().values[0]
    projected_scales.append(yearly_averages.values[-1,0]+mean_grad)

projected_scales = np.array(projected_scales)

# %% Plot absolute profiles compared to test data

cols = test_data[unnormalized_column_group].columns
model_data = np.outer(projected_scales,mean_profile)

# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil(test_data[unnormalized_column_group].shape[1] / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(test_data[unnormalized_column_group].shape[1]):
    ax = axs[i]  # Get the correct subplot
    ax.plot(test_data['date'], test_data[cols[i]], color = 'tab:red',  label='Training Data')
    ax.plot(test_data['date'], model_data[i], label='Model Data', color = 'tab:blue')
    ax.set_title(f'Column: {cols[i]}')  # Set title for each subplot
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure



print(np.sum([(test_data[cols[i]]-model_data[i])**2 for i in range(test_data[unnormalized_column_group].shape[1])]))

#%% The profiles scaled to total sales


def test_r_squared_absolute(
        test_data,
        future_mean_model_predictions,
        model_predictions
        ):
    n_weeks = len(model_predictions)
    unnormalized_column_group = [x for x in test_data.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

    
    n_groups = len(unnormalized_column_group)
    ss_res_raw = np.ones((n_weeks,n_groups))
    ss_tot_raw = np.ones((n_weeks,n_groups))
    for week in range(1,n_weeks+1):
        df_test_week = test_data[test_data['week'] == week][unnormalized_column_group]
        if len(df_test_week) == 0:
            print("No test data")
            return np.nan, np.nan
        if len(model_predictions) == 0:
            print("No profile data")
            return np.nan, np.nan
        ss_res_raw[week-1] = ((df_test_week - model_predictions[week-1])**2).values[0,:]
        ss_tot_raw[week-1] = ((df_test_week - future_mean_model_predictions[week-1])**2).values[0,:]
        
    ss_res = ss_res_raw.mean(axis = 1)
    ss_tot = ss_tot_raw.mean(axis = 1)


    return ss_res, ss_tot

mean_r_squared_absolute = test_r_squared_absolute(
        test_data = test_data,
        future_mean_model_predictions = np.outer(projected_scales,future_mean_model).T,
        model_predictions = np.outer(projected_scales,mean_profile).T
        )

fourier_r_squared_absolute = test_r_squared_absolute(
        test_data = test_data,
        future_mean_model_predictions = np.outer(projected_scales,future_mean_model).T,
        model_predictions = np.outer(projected_scales,fourier_profile).T
        )

abs_mean_r = 1-mean_r_squared_absolute[0]/mean_r_squared_absolute[1]
abs_fourier_r = 1-fourier_r_squared_absolute[0]/fourier_r_squared_absolute[1]

print("mean r-squared mean model: ",abs_mean_r.mean().round(2)," +- ",abs_mean_r.std().round(2))
print("mean r-squared Fourier model: ",abs_fourier_r.mean().round(2)," +- ",abs_fourier_r.std().round(2))


plt.figure()
plt.plot(mean_r_squared_absolute[0], label = "Mean model")
plt.plot(fourier_r_squared_absolute[0], label = "Fourier model")
plt.plot(mean_r_squared_absolute[1], label = "Future mean model")
plt.legend()
