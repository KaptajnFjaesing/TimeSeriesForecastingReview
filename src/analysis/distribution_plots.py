import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm

# Define parameters for the Gamma distribution
# Define your target mean and variance

# mean = alpha/beta
# variance = alpha/beta**2

# alpha = mean*beta
# beta = sqrt(alpha/variance) = sqrt(mean*beta/variance) so beta = mean/variance



hest = 0.4  # Example mean

scale = 1000
variance = hest / scale

# Compute Gamma parameters
alpha = scale * hest
beta = scale

# Define the x values (support for the Gamma distribution)
x = np.linspace(0, 1, 500)

# Calculate the PDF of the Gamma distribution
pdf_gamma = gamma.pdf(x, a=alpha, scale=1/beta)  # scale = 1/beta is used because scipy uses scale = 1/rate

pdf_normal = norm.pdf(x, loc = hest, scale = np.sqrt(hest / scale))

# Plot the Gamma distribution
plt.figure(figsize=(8, 5))
plt.plot(x, pdf_gamma, label=f'Gamma PDF (α={alpha}, β={beta})', color='b')
plt.plot(x, pdf_normal, label=f'Normal PDF (mu={hest}, sigma={hest/scale})', color='r')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Gamma Distribution')
plt.legend()
plt.grid(True)
plt.show()

#%%


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm

# Define parameters for the Gamma distribution
# Define your target mean and variance

# mean = alpha/beta
# variance = alpha/beta**2

# alpha = mean*beta
# beta = sqrt(alpha/variance) = sqrt(mean*beta/variance) so beta = mean/variance




alpha = 2
beta = 3

# Define the x values (support for the Gamma distribution)
x = np.linspace(0, 20, 500)

# Calculate the PDF of the Gamma distribution
pdf_gamma = gamma.pdf(x, a=alpha, scale=1/beta)  # scale = 1/beta is used because scipy uses scale = 1/rate

# Plot the Gamma distribution
plt.figure(figsize=(8, 5))
plt.plot(x, pdf_gamma, label=f'Gamma PDF (α={alpha}, β={beta})', color='b')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Gamma Distribution')
plt.legend()
plt.grid(True)
plt.show()


# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

# Parameters
mu = 0      # Location parameter (mean)
b = 0.5      # Scale parameter (spread)

# Generate x values for the plot
x = np.linspace(mu - 5*b, mu + 5*b, 1000)

# Calculate the Laplace distribution's PDF
pdf = laplace.pdf(x, loc=mu, scale=b)

# Plot the distribution
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, label=f'Laplace Distribution\nμ = {mu}, b = {b}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Laplace Distribution')
plt.legend()
plt.grid(True)
plt.show()
