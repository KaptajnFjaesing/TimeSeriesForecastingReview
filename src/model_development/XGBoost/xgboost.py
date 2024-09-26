import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Step 1: Create a Mock Time Series
np.random.seed(42)
n = 200  # Number of time points
t = np.arange(n)
data = np.sin(t * 0.1) + np.random.normal(scale=0.5, size=n)  # Sine wave with noise

# Convert to a DataFrame
df = pd.DataFrame({'time': t, 'value': data})

# Step 2: Prepare the Data
def create_features(df, n_lags=5, n_forecast=1):
    X, y = [], []
    for i in range(len(df) - n_lags - n_forecast + 1):
        X.append(df['value'].iloc[i:i+n_lags].values)
        y.append(df['value'].iloc[i+n_lags:i+n_lags+n_forecast].values)
    return np.array(X), np.array(y)

n_lags = 10  # Number of lags to use as features
n_forecast = 5  # Number of steps to forecast
X, y = create_features(df, n_lags, n_forecast)

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 3: Train the XGBoost Model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print some of the actual vs predicted values
print("Actual vs Predicted:")
for actual, pred in zip(y_test[:5], y_pred[:5]):
    print(f"Actual: {actual} | Predicted: {pred}")
