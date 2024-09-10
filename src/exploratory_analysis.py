
#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/passengers.csv', parse_dates=['Date'])


plt.plot(df['Date'], df['Value'], marker='o')

plt.title('Monthly Data Over Years')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# %%
