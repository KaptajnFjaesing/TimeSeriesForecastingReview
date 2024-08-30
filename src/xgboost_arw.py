#%% imports
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm

warnings.simplefilter("ignore", category=RuntimeWarning)
#%%
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir,'../data/passengers.csv')
df_passengers = pd.read_csv(data_path)  # Load the data

#%%
df_passengers['Passengers'] = np.log(df_passengers['Passengers']) # Log-scale
df_passengers_0 = df_passengers.iloc[0,:] # save first datapoint
df_passengers['Passengers'] = df_passengers['Passengers'].diff().bfill() # Detrend

context_length = 35 # Define context lenght, past observations to be used as features to the model
train_length = int(len(df_passengers)*0.8) # Training data size
df_features = pd.DataFrame() # New dataframe for storing features

for i in range(1,context_length):
    df_features[f'{i}'] = df_passengers['Passengers'].iloc[:train_length].shift(i) # Fill lagged values as features
df_features.index = df_passengers.iloc[0:train_length].index    
df_features.dropna(inplace=True)

df_features['time_sine']=np.sin(2*np.pi*df_features.index/12) # Create a sine time-feature with the period set to 12 months

y_train = df_passengers.loc[df_features.index]['Passengers'] # Training target data

#%%
model = xgb.XGBRegressor(objective='reg:absoluteerror',
                        n_estimators=100,
                        tree_method='hist',
                        max_leaves=5,
                        max_depth=20,
                        learning_rate=0.1)
model.fit(df_features, y_train)

train_predict = model.predict(df_features.values)
#%% Plot train target and predictions
plt.figure()
plt.plot(y_train.values)
plt.plot(train_predict)

#%% Iterative forecast

forecast_horizon = 26
predictions = np.zeros(forecast_horizon)
context = df_features.values[-1,:]
for i in range(forecast_horizon):
    predictions[i]=model.predict(context[None,:])
    context[1:34] = context[0:33]
    context[0] = predictions[i]
    context[-1] = np.sin(2*np.pi*(y_train.index[-1]+i)/12)

#%% Plot the results (data is still detrended and log scaled)
plt.figure()
plt.plot(df_passengers.index,df_passengers['Passengers'],label='Full actual data')
plt.plot(y_train.index,y_train.values,label='Training data (context=past 35 days)')
plt.plot(y_train.index,train_predict,'.',label='Predicted (train)')
plt.plot(np.arange(y_train.index[-1],y_train.index[-1]+forecast_horizon),predictions,'.-',label='Predicted (test)')
plt.xlabel('Time (months)')
plt.ylabel('Nr. of airpassengers (log scaled, detrended)')
plt.legend()

#%% Plot rescaled data and predictions 
forecast_rescaled = np.exp(np.sum(df_passengers.iloc[0:y_train.index[-1],1])
                            + df_passengers_0['Passengers']
                            + np.cumsum(predictions))
plt.figure()
plt.plot(df_passengers.index,np.exp(np.cumsum(df_passengers['Passengers'])+df_passengers_0['Passengers']),label='Actual')
plt.plot(np.arange(y_train.index[-1],y_train.index[-1]+forecast_horizon),forecast_rescaled,'.-',label='Predicted (test)')
plt.xlabel('Time (months)')
plt.ylabel('Nr. of airpassengers')
plt.legend()

# %%
