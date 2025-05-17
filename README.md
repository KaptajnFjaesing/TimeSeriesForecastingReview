# Python Time Series Forecasting Review

## Introduction

This study represents a collaborative effort to review a diverse set of Python-based time series forecasting modules. The code and project are publicly available on [GitHub](https://github.com/KaptajnFjaesing/TimeSeriesForecastingReview).

The forecast modules are reviewed *relatively out of the box*, meaning individual models from modules have not been given extensive feature engineering or hyper parameter tuning. That said, fast and simple feature engineering has been tested on some models in order to gauge the effect.  The primary objective of the project is to familiarize the Reader with a wide range of Python forecasting modules and provide a preliminary evaluation of their strengths and weaknesses. It is acknowledged that the performance of the models presented here could significantly improve with additional tuning and feature engineering.

## Data
For this study, an aggregated version of the M5 dataset, a benchmark dataset in the forecasting community, has been utilized. The M5 dataset comprises daily sales data from Walmart's stores and departments across various geographical locations in the United States. It is publicly available through the [M5 Forecasting Competition on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/). The dataset's scale, granularity, and hierarchical structure (e.g., store, department, and product categories) make it an ideal testbed for evaluating forecasting algorithms.

To accommodate models with scalability limitations, this study focuses on weekly aggregated sales data for a specific store-category combination, namely household sales. The time range of interest spans from 2011 to 2016 (see Figure 1), with weekly sales data serving as the basis for analysis.

![Weekly Sales for Each Store-Category](./figures/raw_data.png)
_Figure 1: Weekly Sales for Each Store-Category. The training/test split has varied during the analysis (see the Metric section) and is therefore not shown._

## Metric
To quantitatively compare the accuracy of different forecasting algorithms, the Mean Absolute Scaled Error (MASE) is employed. This metric is defined as:

$$
\text{MASE} = \frac{\text{mean absolute forecasting error}}{\text{mean gradient in training data}}.
$$

The denominator represents the average forecasting error when using a naive forecast (predicting the next timestep to be equal to the current one) for a single time step. Thus, MASE provides a relative measure of model performance compared to this baseline. The interpretation should be a bit careful though, since the average gradient only represent the naive forecast a single timestep whereas the models here consider a forecast horizon much longer than that. Hence, intuitively, an excellent model would be better than the naive forecast (MASE < 1) for the first timestep and then gradually get worse (MASE increase over the time horizon).


This study evaluates the simultaneous forecasting of ten time series ($N_{\text{time series}} = 10$) over a forecast horizon of 26 weeks ($N_{\text{forecast horizon}} = 26$). To ensure robust estimates, an ensemble of $N_{\text{ensemble}} = 50$ forecasts is generated. The ensemble is constructed by producing forecasts for each element (henceforth referred to as the reference point) in the interval

$$
[N-(N_{\text{ensemble}} + N_{\text{forecast horizon}}), N - N_{\text{forecast horizon}}]
$$

where $N$ is the number of data points in the time series (all have the same number of data points for simplicity). Stated another way; a forecast consisting of $N_{\text{forecast horizon}}$ steps into the future will be produced for each model. This will be done $N_{\text{ensemble}}$ times while rolling over data. For each time step in the forecast horizon, there will therefore be $N_{\text{ensemble}}$ values from which to determine a MASE for each time series and forecast time step.

Let $f_{i,j,q}$ and $y_{i,j,q}$ denote the model forecast and data, respectively, for time step $i$, time series $j$, and ensemble number $q$, then

$$
\text{MASE}_{i,j} = \frac{
    \frac{1}{N_{\text{ensemble}}} \sum_{q=1}^{N_{\text{ensemble}}} |f_{i,j,q} - y_{i,j,q}|
    }{
        \frac{1}{N_{\text{training}}-1} \sum_{i=1}^{N_{\text{training}}} |y_{i-1,j} - y_{i,j}|
        }
$$
where
$$
N_{\text{training}} \equiv N - (N_{\text{ensemble}} + N_{\text{forecast horizon}}) - 1
$$

represents the part of the data that is common to all ensembles (i.e. where the rolling starts). To condense the information, the MASE can be averaged over time series

$$
\text{MASE}_i \equiv \frac{1}{N_{\text{time series}}} \sum_{j=1}^{N_{\text{time series}}} \text{MASE}_{i,j}
$$
where

$$
\text{MASE}_i \equiv \text{Average MASE over time series}
$$

and again over the forecast horizon

$$
\langle \text{MASE} \rangle \equiv \frac{1}{N_{\text{forecast horizon}} N_{\text{time series}}} \sum_{i=1}^{N_{\text{forecast horizon}}} \sum_{j=1}^{N_{\text{time series}}} \text{MASE}_{i,j}
$$
where for shorthand
$$
\langle \text{MASE} \rangle \equiv \text{Average MASE over time series and forecast horizon}.
$$

As a _rough_ estimate of uncertainty for this average MASE, the standard deviation of the average over time series is considered

$$
\delta \langle \text{MASE} \rangle = \sqrt{\frac{1}{N_{\text{forecast horizon}} - 1} \sum_{i=1}^{N_{\text{forecast horizon}}} (\text{MASE}_i - \langle \text{MASE} \rangle)^2}
$$

This assumes the variation between time series can be neglected relative to the variation between forecast horizon steps. The uncertainty is intended to be a conservative estimate of the variance in the sample, rather than the uncertainty of the mean estimate (if the latter is intended, the variance in equation $\langle \text{MASE}\rangle$ and error propagation can be used).

## Models
This section introduces the models evaluated in this study, highlighting their characteristics, origins, computational approaches, and feature engineering techniques. The models are implemented using a variety of Python packages, including `statsmodels`, `darts`, `gluonts`, `sorceror`, and custom implementations. 

### Holt-Winters
The **Holt-Winters** model, implemented in the `statsmodels` package, is a state space model designed to capture additive seasonality and trend components in time series data. Min-max normalization is applied to the training data to scale the values between 0 and 1. The model is trained on the normalized data, and forecasts are denormalized using the inverse of the normalization.

### State Space Model (SSM)
The **State Space Model (SSM)**, also from `statsmodels`, leverages state space representations to model time series. It uses components such as autoregressive terms, level, trend, and seasonal harmonics. Similar to Holt-Winters, min-max normalization is applied to the training data, and forecasts are denormalized similar to the Holt-Winters model.

### Sorcerer
The **Sorcerer** model, from the `sorcerer` package, is a hierarchical Bayesian Generalized Additive Model (GAM) for time series forecasting, inspired by timeseers and the PyMC model builder class. It utilizes information across time series to produce predictions for each time series. Detailed information regarding individual and shared trends and seasonalities can be extracted from the model. Although the model takes the raw traning and test data as input, it implicitly apply min-max normalization.

### TiDE
The **TiDe** model, available in the `darts` package, combines temporal convolutional networks and recurrent neural networks for flexible time series forecasting. No feature engineering is applied for this model.

### Naive

The **Naive** models, implemented in `darts`, rely on straightforward extrapolation techniques to generate forecasts. No feature engineering is applied for these models. They serve as baseline approaches, offering simple yet effective methods for comparison with more complex forecasting algorithms.

- The **Naive Drift** model

$$
\hat{y}_{t+h} = y_{t}+h\frac{y_{t_{n}}-y_{t_{0}}}{t_{n}-t_{0}},
$$

where $t_{n}$ and $t_{0}$ represent the end and start of training data, respectively. 

- The **Naive Seasonal** model

$$
\hat{y}_{t+h} = y_{t + (h \mod K) - K},
$$

where $K$ is the seasonal lag.

- The **Naive** model is equal to the Naive Seasonal model with $K=1$. It repeats the last observation $y_t$ for the entire forecasting horizon $\hat{y}_{t+1}, \hat{y}_{t+2},\dots \hat{y}_{t+N_{\text{forecast horizon}}}$.

### Light GBM
The **LightGBM** model is a tree-based model for regression and classification tasks. In time series forecasting, it can be used to predict future values by learning patterns from e.g. lag features, datetime features and exogenous variables. The model is evaluated in multiple configurations;

- The 'basic implementation' utilize the `lightgbm` package and apply a log transformation, differencing, lagged differences and sine-based time as features. 
- The 'sklearn implementation' utilize the `lightgbm` and `sklearn` packages, the latter is used to determine the hyperparameters via a grid search, with "day_of_year", "month", "quarter", "year" as features.
- The 'darts implementation' utilize the `darts` package and no feature engineering.
- The 'feature darts implementation' utilize the `darts` package and apply a log transformation, differencing, lagged differences and sine-based time as features.

### XGBoost
The **XGBoost** model is a regularized version of gradient-boosted decision trees. It works similarly to LightGBM but includes additional techniques like regularization and tree pruning. It is available in the `darts` package and is used without feature engineering.

### TFT
The **Temporal Fusion Transformer (TFT)** model, available in the `darts` package, is a probabilistic deep learning model designed for interpretable forecasting. Feature engineering consists of a log transformation, differencing, lagged differences and sine-based time as features.

### Climatological
The **Climatological** model, available in the `darts` package, predicts future values by taking the mean of random samples from the historical data, assuming that future values follow the same distribution as past values. This approaches a constant value that equals the mean of the historical data with noise dependent on the number of random samples.

### Mean
The mean approach is based on normalized seasonal profiles. These are generated by normalizing observations by the yearly mean 

$$
\text{YearlyMean}_{j} = \frac{1}{N_{j}} \sum_{s=1}^{N_{j}} y_{j,s}
$$

and computing the average for each week. Let $y_{j,w}$ denote a datapoint for year $j$ and week $w$, then the normalized observations can be written

$$
y_{j,w}^{\text{normalized}} = \frac{y_{j,w}}{\text{YearlyMean}_{j}}
$$

where $N_{j}$ denotes the observations for year $j$. The seaonality profile is given by

$$
\text{SeasonalityProfile}_{w} = \frac{1}{N_{w}} \sum_{j=1}^{N_{w}} y_{j,w}^{\text{normalized}},
$$

where $N_{w}$ is the number of years with data for week $w$. Forecasts are generated via multiplying the seasonality profile with a scale

$$
\hat{y}_{t+h} = \text{scale}_{t}\cdot\text{SeasonalityProfile}_{t+h},
$$ 

where the scale depends on the method.

- The **Static Mean** model computes the scale as follows

$$
\text{scale}_t = \text{YearlyMean}_{\text{most recent year}} + \text{mean gradient of 'YearlyMean'},
$$

which gives a constant scale that accounts for both the current level and a simple trend.

- The **Rolling Mean** model computes the scale as the mean of the last $n$ observations as the forecast

$$
\text{scale}_t = \frac{\text{mean $y$ in last n observations at time $t$}}{\text{SeasonalityProfile}_{t-\frac{n}{2}}},
$$

the scale consists of the rolling mean normalized by the seasonality profile evaluated at the middle of the rolling window, $n$. This makes the forecast more responsive to recent trends in the data.

### DeepAR
The **DeepAR** model, implemented in the `gluonts` package, uses autoregressive recurrent neural networks to generate probabilistic forecasts. Feature engineering includes log transformation and differencing. Rolling windows are used for training, validation, and testing, with the splits defined as $y_{1:N-2\cdot\text{fh}}$ for training, $y_{N-2\cdot\text{fh}:N-\text{fh}}$ for validation, and $y_{N-\text{fh}:N}$ for testing. Predictions are back-transformed to the original scale.

### TLP
The **Two-Layer Perceptron (TLP)** is a Bayesian neural network implemented using `pymc`. Feature engineering includes sine-based time features and lagged differences, with min-max normalization applied to the training data. The model is trained using Bayesian inference, and probabilistic forecasts are generated and denormalized.

## Results
Figure 2 provides a summary of the average MASE over time series and forecast horizon for each model. The second column ranks the models based on their performance, with the best-performing models listed at the top. The third column illustrates the computation time for each model relative to the longest computation time (Sorcerer with NUTS). 

![Average MASE Over Time Series and Forecast Horizon](./figures/model_summary_table.png)
_Figure 2: Summary table._

Figure 3 illustrates the average MASE over time series for each forecast horizon step. This figure provides a more granular view of how the models perform as the forecast horizon progresses. The second column in Figure 2 is derived by averaging the values shown in Figure 3.

![Average MASE Over Time Series](./figures/avg_mase_subplots_three_columns.png)
_Figure 3: Average MASE Over Time Series._

## Discussion
The results reveal several key insights into the performance of the forecasting models:

1. **Performance Degradation Over Time**: As expected, the models exhibit a decline in performance as the forecast horizon extends. This serves as a sanity check, confirming that the models behave as anticipated when tasked with predicting further into the future. The degradation highlights the increasing uncertainty and difficulty associated with long-term forecasting.

2. **Wide Performance Variability**: The models demonstrate a broad range of performance levels. Notably, only the Holt-Winters and State Space Models (SSM) consistently outperform the average gradient in the training data for the first time step in the forecast horizon. However, this superior performance is limited to relatively short forecast horizons, which aligns with expectations. This finding underscores the inherent difficulty of the forecasting problem, making it an ideal testbed for distinguishing between models. The variability also highlights the importance of selecting models that align with the specific characteristics of the dataset and forecasting task.

3. **Surprising Strength of Naive Models**: The Naive models perform surprisingly well, further emphasizing the challenging nature of the dataset. Their strong performance suggests that the underlying signal in the data is difficult to predict consistently. Given the definition of MASE, it is expected that the 'Naive Darts' model would have MASE ≈ 1 for the first timestep in the forecast horizon. While this is somewhat the case, the slight deviation from 1 at the first timestep illustrates a gradual increase in the average gradient of the data over time.

4. **Challenge of Long-Horizon Forecasting**: The primary value of the forecasting models lies not in predicting a single timestep ahead but in providing insights across the entire forecast horizon. Given the difficulty of the dataset, achieving a MASE below 1 at any time step in the forecast horizon is challenging (see Figure 2). Consistently achieving a MASE below 1 further into the forecast horizon is even more difficult. The MASE metric, which benchmarks against the average gradient of the training data, is intentionally stringent. Consequently, MASE values below 1 are realistically attainable only for the initial portion of the forecast horizon. This highlights the importance of evaluating models not just on their short-term accuracy but also on their ability to maintain performance over longer horizons.

5. **Insights into Model Selection**: The findings underscore the importance of selecting appropriate models for specific forecasting tasks. While larger, more complex models may excel in scenarios with rich, complex signals and abundant data, simpler models often outperform them when these conditions are not met. 

6. **Interpretation of Results**: It is important to note that the purpose of this study is not to provide a definitive ranking of the models but rather to familiarize readers with their "out of the box" performance. As stated in the introduction, the performance of the models presented here could significantly improve with additional tuning and feature engineering. The results should therefore be interpreted as a preliminary evaluation rather than a final judgment of the models' capabilities.

These findings highlight the complexity of the forecasting problem and the importance of utilizing a range of benchmark models to gauge the performance of more complex algorithms. The study serves as a valuable resource for understanding the strengths and weaknesses of various forecasting approaches and provides a foundation for further exploration and refinement. Via the open-source code, the project also provides a template for how the models can be used in practice.