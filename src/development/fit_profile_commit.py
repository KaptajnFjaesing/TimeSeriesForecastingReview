
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic monthly data
np.random.seed(41)
T = 52
intervals = np.array(range(1, T+1))  # Months from 1 (January) to 12 (December)
years = 5  # Number of years of data
data = []

# Create a sinusoidal pattern with some noise
rate = 10 + 5 * np.sin(2 * np.pi * intervals / T)
for year in range(years):
    data.extend(np.random.poisson(lam=rate, size=intervals.shape))

x_train = np.tile(intervals, years)
y_train = np.array(data)


plt.figure()
plt.plot(intervals,rate)
plt.plot(x_train,y_train, 'o')
plt.xlabel('Week')
plt.ylabel('Value')

# %%
n_terms = 4
T = 52

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt


def construct_pymc_model(x_train: np.array, y_train: np.array):
    with pm.Model() as model:     
        param_a = pm.Gamma('param_a', alpha=2.4, beta=3)
        param_b = pm.Gamma('param_b', alpha=2.4, beta=3, shape = (n_terms,))
        param_c = pm.Gamma('param_c', alpha=2.4, beta=3, shape = (n_terms,))
        
        K = np.arange(1, n_terms + 1)
        
        cos_terms = pm.math.cos(2 * np.pi * np.outer(x_train, K) / T)
        sin_terms = pm.math.sin(2 * np.pi * np.outer(x_train, K) / T)
        
        b_terms = pm.math.dot(cos_terms, param_b)
        c_terms = pm.math.dot(sin_terms, param_c)
        
        lambda_ = pm.Deterministic('lambda_', param_a + b_terms+c_terms)
        
        mu = pm.math.exp(lambda_)
        y = pm.Poisson('y', mu = mu, observed = y_train)
        
    return model

model = construct_pymc_model(x_train = x_train, y_train = y_train)


with model:
    posterior = pm.sample(tune=1000, draws=5000, chains=1)


param_a = posterior.posterior['param_a'].mean((('chain'))).values
param_b = posterior.posterior['param_b'].mean((('chain'))).values
param_c = posterior.posterior['param_c'].mean((('chain'))).values


def iteration1(a,b,c,n_terms,T,weeks):
    K = np.arange(1,n_terms+1)
    return np.exp(a+np.sum(b*np.cos(2*np.pi*np.outer(weeks,K)/T)+c*np.sin(2*np.pi*np.outer(weeks,K)/T), axis = 1))


# %%
# Create a high-resolution time axis for plotting, including a wrap-around point
high_res_weeks = np.linspace(1, T+1, 1000)  # Go slightly beyond 12 to ensure smooth wrapping

preds = np.zeros((len(high_res_weeks),len(param_a)))
for i in range(len(param_a)):
    preds[:,i] = iteration1(param_a[i],param_b[i],param_c[i],n_terms,T,high_res_weeks)

plt.figure()
plt.plot(intervals,rate)
for i in range(len(param_a)):
    plt.plot(high_res_weeks, preds[:,i], alpha = 0.01, color = 'red')
    
plt.plot(high_res_weeks, preds.mean(axis = 1), color='black', label='Fitted Model Bayesian', linewidth=2)
plt.plot(x_train,y_train, 'o')
plt.xlabel('Week')
plt.ylabel('Value')

# Convert months to angles (theta) for polar plot
theta = 2 * np.pi * (x_train - 1) / T
theta_high_res = 2 * np.pi * (high_res_weeks - 1) / T

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')
ax.scatter(theta, data, color='blue', label='Data', alpha=0.6)
for i in range(len(param_a)):
    plt.plot(theta_high_res, preds[:,i], color='red', alpha=0.01)  # Set alpha to control opacity
    
ax.plot(theta_high_res, preds.mean(axis = 1), color='black', label='Fitted Model Bayesian', linewidth=2)
ax.set_theta_direction(-1)  # Optional: Reverse the direction of the plot (clockwise)
ax.set_theta_offset(np.pi / 2.0)  # Optional: Set 12 o'clock (January) as the top of the circle
ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_title('Polar Plot of Time Series with Continuous and Smooth Fourier Series Model', va='bottom')
plt.legend(loc='upper right')

