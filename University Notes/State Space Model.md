[[Time Series]]

Notes from time series analysis book: [[time_series_analysis.pdf]]

In general a **State Space Model** is divided in two main parts

1. There is an hidden or latent process $x_t$ called ***state process***, this is assumed to be a Markov Process, meaning that future $x_{t+1}$ and past observations $x_{t-1}$ are *conditionally independant* given the present $x_t$
2. The second part is that the observations $y_t$ are independent given the states $x_t$, this means that the dependance relation among observation is generated by states ($x$). 

Be careful because the markov  assumption usually do not holds for state space models (differential equation depends on the past obervations)

![[Pasted image 20240118164401.png]]

This state space models resmbles very much [[Lesson 7 - Hidden Markov Model]] we saw in the ISPR course

Main differences between these two models are the following

1. SSMs model structure consists of two sets of equations, *state equation* (how state evolves) and *observation equation* (how hidden state generates observable data) these two functions can be non linear. Instead in HMMs we have a set of transition probabilities between states and emission probabilities for each state so much more discrete variant..
2. The second point is the Continuous vs Discrete in SSM we can have both continuous or discrete observations while HMM is only a discrete probabilistic approach.
3. The fact that usually SSM do not assume a markov process underlying the data.

## Linear Gaussian Model

This model in it's basic form use the following as **state equation** p-dimensional vector autoregression

$$
x_t = \phi_{t-1}+w_t
$$

$w_t$ are $p \times 1$ i.i.d, zero mean normal vectors with covariance matrix $Q$: $w_t \approx iid N_p(0,Q)$.

We assume the process starts with a normal vetor $x_0$ such that $x_0 \approx N_p(\mu_0,\Sigma_0)$. We do not observe the state but only a *linear transformation* of it:

$$
y_t = A_tx_t+v_t
$$

Where $A_t$ is a $q \times p$ measurement or observation matrix and the latter equation is **observation equation**. This observed data vector $y_t$ is q-dimensional (can be larger than the input p), and then we add that noise $v_t$, uncorrelated with the $w_t$ noise.

## Filtering, Smoothing and Forecasting

Any analysis on SSM would be to produce estimators for the unobserved signal $x_t$ given data $y_{1:s}$ to time s, state estimation is essential component of parameter estimation, if $s < t$ the problem is called forecasting since we want to estimate the future given past observations, instead when $s = t$, we call it filtering because we want to predict the present state given the past observations ($p(x_t | y_1,...,y_t)$). Then we have the smoothing when $s > t$ so when we want to get ($p(x_s | y_1,...,y_t)$).

Let's define 

$$
x_t^s = E(x_t | y_1:s)
$$
and 

![[Pasted image 20240118175243.png]]

in this case the P is the corresponding mean squared error
## Kalman Filter

Given the SSM we saw earlier we define the following

the timesteps goes from $t=0$ to $t=n$.

![[Pasted image 20240118175405.png]]

The last part is the *Kalman Gain*. Prediction for $t>n$ is accomplished using the first two equations