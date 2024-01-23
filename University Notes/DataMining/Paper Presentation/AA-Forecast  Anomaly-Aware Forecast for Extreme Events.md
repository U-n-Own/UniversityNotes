[[Machine Learning]], [[Statistics]], [[Stability]], [[Gaussian Process]], [[Monte Carlo Dropout]], [[Time Series]], [[Anomaly Detection]], [[Lesson 11 - Attention and Transformers#Attention]]
# Problem formulation

# Model : AA-Forecast

## Decomposition

## Anomaly-Aware model

## Dynamical Uncertainty Optimization

The things here is that uncertainty has to be minimized to do prediction that are pretty near the real outcome, this is difficult. What they did is apply [[Monte Carlo Dropout]] and getting $M$ forecasts for each timestep $t$ in an online fashion from trained model then obtaining these $M$ output $y*$ and averaged them.

$$
\bar{y_*} = \frac{1}{M}\sum_{m=1}^M y_{(m)}*
$$

The *Uncertainty* is the variability of (prediction) distribution, the *stdev* (SD) of the distribution of future observations conditional on the information avaiable at forecast time.

How to derive optimal dropout probability $p$ at each timestep? 

We add increments of $0.1$, for $p=0$ the model forecasting

