from src.functions import load_passengers_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
import warnings


warnings.simplefilter("ignore", category = RuntimeWarning)


df_passengers = load_passengers_data()  # Load the data

# Feature engineering
df_passengers['Passengers_log'] = np.log(df_passengers['Passengers']) # Log-scale
df_passengers_0 = df_passengers.iloc[0,:] # save first datapoint
df_passengers['Passengers_log_diff'] = df_passengers['Passengers_log'].diff()#.bfill() # Detrend

context_length = 35 # Define context lenght, past observations to be used as features to the model
train_length = int(len(df_passengers)*0.8) # Training data size
df_features = pd.DataFrame() # New dataframe for storing features

for i in range(1,context_length):
    df_features[f'{i}'] = df_passengers['Passengers_log_diff'].iloc[:train_length].shift(i) # Fill lagged values as features
df_features.index = df_passengers.iloc[0:train_length].index    
df_features.dropna(inplace=True)

df_features['time_sine'] = np.sin(2*np.pi*df_features.index/12) # Create a sine time-feature with the period set to 12 months

"""
I want the first test input to be an input the algorithm has not seen before,
so I hold the last point for this.

Note: Index has been constructed such that input and output match.
Note: I do not use bfill(), so my df_features is shifted one index due to drop_na


"""

df_features_hold_out = df_features.iloc[:-1]
y_train = df_passengers.loc[df_features_hold_out.index]['Passengers_log_diff'] # Training target data


print(y_train)
#%%


# Create LightGBM datasets
train_dataset = lgb.Dataset(
    data = df_features_hold_out.values,
    label = y_train.values
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

train_predict = model.predict(df_features_hold_out.values)
#%% Plot train target and predictions
plt.figure()
plt.plot(y_train.values)
plt.plot(train_predict)

#%% Iterative forecast

"""
The first index for test data must be the hold one out.
I do not have an overlap between test and training data due to the holdout.

"""

forecast_horizon = 26
predictions = np.zeros(forecast_horizon)
context = df_features.iloc[-1].values
test_start_index_plus_one = df_features.index[-1]+1

for i in range(forecast_horizon):
    predictions[i] = model.predict(context[None,:])
    context[1:34] = context[0:33]
    context[0] = predictions[i]
    context[-1] = np.sin(2*np.pi*(test_start_index_plus_one+i)/12) # this is for next iteration


#%% Plot the results (data is still detrended and log scaled)
plt.figure()
plt.plot(df_passengers.index,df_passengers['Passengers_log_diff'],label='Full actual data')
plt.plot(y_train.index,y_train.values,'-', label='Training data (context=past 35 days)')
plt.plot(y_train.index,train_predict,'.',label='Predicted (train)')
plt.plot(np.arange(test_start_index_plus_one-1,test_start_index_plus_one-1+forecast_horizon),predictions,'.-',label='Predicted (test)')
plt.xlabel('Time (months)')
plt.ylabel('Nr. of airpassengers (log scaled, detrended)')
plt.legend()

#%% Plot rescaled data and predictions 
forecast_rescaled = np.exp(
    df_passengers_0['Passengers_log']
    + np.sum(df_passengers['Passengers_log_diff'].iloc[1:test_start_index_plus_one-1])
    + np.cumsum(predictions)
    )
                        
plt.figure()
plt.plot(df_passengers.index,np.exp(np.cumsum(df_passengers['Passengers_log_diff'])+df_passengers_0['Passengers_log']),label='Actual')
plt.plot(np.arange(test_start_index_plus_one-1,test_start_index_plus_one-1+forecast_horizon),forecast_rescaled,'.-',label='Predicted (test)')
plt.xlabel('Time (months)')
plt.ylabel('Nr. of airpassengers')
plt.legend()

# Results are comparable.