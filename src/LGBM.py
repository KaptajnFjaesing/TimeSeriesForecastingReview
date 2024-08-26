#%% setup data
from src.functions import load_passengers_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_passengers = load_passengers_data()  # Load the data


def create_lagged_features(df, column_name, n_lags, n_ahead):
    """Create lagged features for a given column in the DataFrame, with future targets."""
    lagged_df = pd.DataFrame()
    
    # Create lagged features
    for i in range(n_lags, -1, -1):
        lagged_df[f'lag_{i}'] = df[column_name].shift(i)
    
    # Create future targets
    for i in range(1, n_ahead + 1):
        lagged_df[f'target_{i}'] = df[column_name].shift(-i)
    
    lagged_df.dropna(inplace=True)  # Remove rows with NaN values due to shifting
    return lagged_df

# Parameters
n_lags = 35  # Number of lagged time steps to use as features
forecast_horizon = 26  # Number of future time steps to predict

# Create lagged features and future targets
lagged_features = create_lagged_features(df_passengers, 'Passengers', n_lags = n_lags, n_ahead = forecast_horizon)

# Split the data
train_size = int(len(lagged_features) * 0.6)  # 60% for training, 40% for testing
training_data = lagged_features.iloc[:train_size]
test_data = lagged_features.iloc[train_size:]


normalization = 400
# Define features and targets
x_train = training_data[training_data.columns[:n_lags+1]]/normalization  # Lagged features
y_train = training_data[training_data.columns[n_lags+1:]]/normalization  # Future targets

x_test = test_data[test_data.columns[:n_lags+1]]/normalization  # Lagged features
y_test = test_data[test_data.columns[n_lags+1:]]/normalization  # Future targets

if len(x_test)<forecast_horizon:    
    print("WARNING: Forecast_horizon too long relative to test/training split")
    
#%%

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.simplefilter("ignore", category=RuntimeWarning)


"""
It appears as if Light GBM only can be used to perform one-step forecasting.
I would like to perform a 26 step forecasting, starting one point beyond what
the model has seen during training;

x_test.iloc[forecast_horizon:forecast_horizon+1]

To do this, I train a new model for each target in the forecast, with the same
input. Intuitively, this goes something like 
    
    for i in range(forecast_horizon):
    - "given training data, predict i ahead".
    

data (X): A 2D array-like structure (e.g., NumPy array or Pandas DataFrame) where each row represents a sample, and each column represents a feature.
Labels (y): A 1D array-like structure (e.g., NumPy array or Pandas Series) where each element corresponds to the label or target value for a sample.


"""

# List to hold models and predictions
predictions = []
# Train one model per forecast step
for i in tqdm(range(forecast_horizon)):
    # Create LightGBM datasets
    train_dataset = lgb.Dataset(
        data = x_train,
        label = y_train.iloc[:,i:i+1]
        )
    
    # Define model parameters
    params = {
        'verbose': -1,
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 5,
        'learning_rate': 0.1,
        'feature_fraction': 0.9
    }
    
    # Train the model with early stopping
    model = lgb.train(
        params,
        train_dataset, 
        num_boost_round = 2000
        )
    predictions.append(model.predict(x_test.iloc[forecast_horizon:forecast_horizon+1], num_iteration=model.best_iteration))

# Convert the list of predictions to a DataFrame
preds_df = pd.DataFrame(np.array(predictions).T, columns=[f'forecast_{i+1}' for i in range(forecast_horizon)])*normalization

# Denormalize actual test targets
y_test_denorm = y_test*normalization

#%%
# Visualize the results
N_train = x_train.shape[1]+x_train.shape[0]-2
N_test = len(df_passengers)-N_train

cut = N_train+forecast_horizon+2

plt.figure(figsize = (20,10))
plt.plot(np.arange(N_train+1),df_passengers['Passengers'].iloc[:N_train+1], label = "Training Data")
plt.plot(np.arange(cut,cut+forecast_horizon),preds_df.iloc[0].values, label ="Light GBM")
plt.plot(np.arange(N_train,N_train+N_test),df_passengers['Passengers'].iloc[N_train:], label = "Test Data", color = "Green")
plt.ylabel("Passengers", fontsize=18)
plt.xlabel("Month", fontsize=18)
plt.xticks( fontsize=18)
plt.yticks( fontsize=18)
plt.minorticks_on()
plt.grid(which='both', linestyle='--', linewidth=0.5)  # Customize grid appearance
plt.legend(loc='upper left',fontsize=18)
