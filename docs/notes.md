# Notes

- pymc_perceptron: With {activation = 'relu' ,n_hidden_layer1 = 10, n_hidden_layer2 = 10} you get a lot of divergences and slow training. The divergences can be reduced by increasing model complexity  e.g. {activation = 'relu' ,n_hidden_layer1 = 20, n_hidden_layer2 = 20}.

- The swish adaptation did not work as intended. It worked fine for the pm.sample_posterior_predictive method, but not the manual coefficient method. The methods converged when changing from x*pm.math.sigmoid(x) to x/(1+np.exp(x)). This indicates that the sigmoid definition for pymc does not follow the convention.

- I changed the gamma prior parameter to be less restrictive; alpha = beta = 1 seems better, such that the prior is peaked towards 0 and has a long tail. A small precision means a large variance...