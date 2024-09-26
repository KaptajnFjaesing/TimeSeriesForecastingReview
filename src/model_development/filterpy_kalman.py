# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:33:29 2024

@author: roman
"""



import numpy as np
from src.functions import load_passengers_data
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter



"""

I have a folder structure in my project on the form

project 
- src
- data

This file is located in src. I want to import functions from src also, and I want the root
folder to be project, such that my import statement looks like this:

from src.functions import load_passengers_data
import sys
from src.functions import load_passengers_data

How do I do this?

sys.path.insert(0, '/path/to/project')


"""
# Load your data
df_passengers = load_passengers_data()  # Replace with your data loading function

normalization = 1

training_split = int(len(df_passengers)*0.7)

x_train = df_passengers.index[:training_split].values/normalization
y_train = df_passengers['Passengers'].iloc[:training_split]/normalization


x_test = df_passengers.index[training_split:].values/normalization
y_test = df_passengers['Passengers'].iloc[training_split:]/normalization

# %%

# Initialize Kalman Filter
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([0, 0])  # Initial state (location and velocity)
kf.P *= 100.  # Initial uncertainty
kf.F = np.array([[1, 1],
                 [0, 1]])  # Transition matrix
kf.H = np.array([[1, 0]])  # Measurement matrix
kf.R = 5  # Measurement noise
kf.Q = np.array([[1, 0],
                 [0, 1]])  # Process noise

# Apply the Kalman filter to the data
filtered_state_means = []
filtered_state_covariances = []

for z in y_train:
    kf.predict()
    kf.update(z)
    filtered_state_means.append(kf.x[0])
    filtered_state_covariances.append(kf.P[0, 0])

filtered_state_means = np.array(filtered_state_means)
filtered_state_covariances = np.array(filtered_state_covariances)

#%%

# Step 3: Define variables for forecasting
n_steps_ahead = len(x_test)  # Number of steps to forecast
last_state = kf.x.copy()  # Initial state for forecasting, after fitting to training data
last_covariance = kf.P.copy()  # Initial covariance, after fitting to training data

# Lists to store predictions and uncertainties
predictions = []
uncertainties = []

# Step 4: Forecast N steps into the future
for _ in range(n_steps_ahead):
    # Predict the next state and covariance
    last_state = kf.F.dot(last_state)
    last_covariance = kf.F.dot(last_covariance).dot(kf.F.T) + kf.Q

    # Store the mean prediction (first element of the state vector) and its uncertainty
    predictions.append(last_state[0])  # State estimate (position or value)
    uncertainties.append(np.sqrt(last_covariance[0]))  # Uncertainty in the estimate

# Convert predictions and uncertainties to numpy arrays
predictions = np.array(predictions)
uncertainties = np.array(uncertainties)


#%%

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(x_train, y_train, label="Actual")
plt.plot(x_test, y_test, label="Actual", color="blue")
plt.plot(x_train, filtered_state_means, label="Kalman Filter Estimate", color="orange")
plt.plot(x_test, predictions, label="Forecast", color="red")
plt.fill_between(
    np.arange(len(y_train), len(y_train) + n_steps_ahead),
    predictions - 1.96 * uncertainties[:,0],
    predictions + 1.96 * uncertainties[:,0],
    color="red",
    alpha=0.3,
    label="95% CI"
)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.title("Kalman Filter Forecasting")
plt.show()
