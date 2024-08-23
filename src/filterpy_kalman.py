

import numpy as np
from src.functions import load_passengers_data
import pandas as pd
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

# %%
# Convert DataFrame to numpy array
data = df_passengers['Passengers'].values

# Initialize Kalman Filter
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([0, 0])  # Initial state (location and velocity)
kf.P *= 1000.  # Initial uncertainty
kf.F = np.array([[1, 1],
                 [0, 1]])  # Transition matrix
kf.H = np.array([[1, 0]])  # Measurement matrix
kf.R = 5  # Measurement noise
kf.Q = np.array([[1, 0],
                 [0, 1]])  # Process noise

# Apply the Kalman filter to the data
filtered_state_means = []
filtered_state_covariances = []

for z in data:
    kf.predict()
    kf.update(z)
    filtered_state_means.append(kf.x[0])
    filtered_state_covariances.append(kf.P[0, 0])

filtered_state_means = np.array(filtered_state_means)
filtered_state_covariances = np.array(filtered_state_covariances)

# Forecasting N steps ahead
n_steps_ahead = 10
predictions = []
last_state = kf.x.copy()

for _ in range(n_steps_ahead):
    last_state = kf.F.dot(last_state)  # State prediction
    predictions.append(last_state[0])

# Generate forecast dates
forecast_dates = pd.date_range(df_passengers['Date'].iloc[-1], periods=n_steps_ahead+1, freq='M')[1:]

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df_passengers['Date'], df_passengers['Passengers'], label="Actual", color="blue")
plt.plot(df_passengers['Date'], filtered_state_means, label="Kalman Filter Estimate", color="orange")
plt.plot(forecast_dates, predictions, label="Forecast", color="red")
plt.fill_between(df_passengers['Date'], 
                 filtered_state_means - 1.96 * np.sqrt(filtered_state_covariances),
                 filtered_state_means + 1.96 * np.sqrt(filtered_state_covariances),
                 color='orange', alpha=0.2)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.title("Kalman Filter Forecasting")
plt.show()
