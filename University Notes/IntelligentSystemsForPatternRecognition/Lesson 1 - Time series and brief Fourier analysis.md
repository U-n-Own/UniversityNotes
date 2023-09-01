[[2023-02-26]]

Status: #notes 

Tags: [[Ispr]], [[Time Series]], [[Signal Processing]], [[Probability]], [[A.I. Master Degree @Unipi]]


Slides from professor of this lesson can be found [here](https://elearning.di.unipi.it/pluginfile.php/54174/mod_resource/content/3/ISPR_2_timeseries.pdf)
Generative and probabilistic models generally are used to learn probability distribution of our data and then drawing samples from them to generete new ones.

## Pattern recognition

Some history: Duda and Hart are like founders, in that time there were't references to learning. Viola-Jones algorithm used for face recognition, based on filters, at the time they looked to 5k hand aligned examples!
Now we're doing essentially data 'matchers'. Usefulness of processing sequences is due to rescuing gradients.

## Time Series

Signals that change in time from observation that are produced from a stochastic process. For example sensors, stock market, etc. Sample can be irregularly spaced out or reguarly spaced, based on what we are working on. One can change from irregularly sampled to regularly sampled with some techniques or can change the model.

A time series $x$ is a sequence of measurements in time $t$. Time series analysis assume weakly stationarity.

*Expectation* don't change with time : $\mathbb{E}[x_t]=\mu \;\; \forall t$ 

If you have a non stationary time series:

Introduce a lag $\tau$ the *Covariance* is $Cov[x_t x_{t+\tau}]= \gamma_\tau \;\; \forall t$ so the covariance between two observations doesn't change with time, without considering the lag.

Covariance is defined as follows

$$
Cov[x_t,x_{t+\tau}]=\mathbb{E}[(x_{t}-\mathbb{E}[x_{t}])(x_{t+\tau} - \mathbb{E}[x_t+\tau])
$$

This measure says that for big values of the first variable $X$ corresponds big values of $Y$ and the same for smaller values, then Covariance is $\gt 0$, othwerwise is negative, so basically show the *tendency of a linear relationship* between $X,Y$. If is zero they do not vary togheter

### Goals for Time Series

What does means to analize time series? Is there any pattern that we can find across the time, can i use the past to predict the future?  Can i control a time series, so adjusting parameters to make it fit a target. eg. Robot movements in a continuous space.

Some common techniques are: 

- ***Baseline*** Is the problem simple, complex... the performance and other.
- ***Preprocessing*** Is the data clean? Do we need to do some preprocessing?  

Time analysis works with **Correlation, Convolution and Autoregressive** models.
Then we want to know the frequences with **Spectral domain analysis**. Later on in the course we can study bot **time** and **frequence**.
  
### Means and Autocovariance
Now we can see some estimators for time series like

Sample mean:
$$
\bar{x}=\frac{1}{N}\sum_{t=1}^N x_t
$$

**Sample Autocovariance** for a lag $-N \leq \tau \leq N$ : is computed at specifig lag of the signal (our input), so you take observations subtract means and then multiply them. 

$$
\hat{\gamma}_x(\tau) = \frac{1}{N} \sum_{t=1}^{N-|\gamma|} (x_{t+|\tau|} - \bar{x})(x_{t} - \bar{x})
$$

We can use autocovariance to compute ***Autocorrelation***: the correlation with the signal with itself.
This is the autocovariance normalized by autocovariance without delay.

$$
\rho_x(\tau)=\frac{\gamma_x(\tau)}{\gamma_x(0)}
$$

[Autocorrelogram]
![[IntelligentSystemsForPatternRecognition/images/Pasted image 20230226160052.png]]

If two points are higly correlated, they are close to each other. Autocorrelation gives us a picture of the signal is repeating itself. If we evaluate the autocorrelation at certain lags $\tau_i$ we can get an autocorrelation plot. These will tells us the strong periodicity of the signal. Near the zero we have the trivial case, the autocorrelation is 1.

Autocorrelation goes from $[-1,1]$, when it's positive we have that negative changes at time t results in negative changes at time $t+\tau$.

### Two time series

Cross-correlation (discrete), a measure of two time series and how they are similar to each other take $x_1$ and $x_2$ as two distinct time series.
The following is our ***measure of similarity***:

$$
\phi_{x1,x2}(\tau) = \sum_{t=max{[0,\tau]}}^{min{[(T^1-1+\tau),(T^{2}-1)]}} x_1(t-\tau)x_{2}(t)
$$
where
$$
 \tau \in [-T^{1+1},..., 0 ,...T^2-1]
$$$$
 \max{\phi_{x_1,x_{2}(\tau)}} \; wrt \; to \; \tau: Dispalacement \; x_{1} \; vs \; x_{2}
$$
We can take the normalized version of $\phi$ to get the cross-correlation, if this is 1 the series have exactly the same shape. Aligning them at the same $\tau$, at $-1$ you have the inverse one of the other, same shape but opposite sign (anticorrelated). At zero they are completely incorrelated (in a linear word), no linear dependancies, but there may be non-linear dependancies to take into account. Speaking linearly you can say that!

**Normalized displacement** return a values that is invariant wrt amplitude

$$
\bar{\phi}_{x1,x2}(\tau) = \frac{\phi_{x1,x2}}{\sum_{t=0}^{T^1-1}(x^1(t))^2\sum_{t=0}^{T^2-1}(x^2(t))^2} \in [-1,1]
$$

### Convolution

Well cross correlation resemble really like [[convolution]].

$$
(f*g)[n]=\sum_{t=-M}^{M} f(n-t)g(t)
$$

Convolution is a symmetric behaviour of the signal, if you cross-correlate two signal you get different results, so cross correlation isn't commutative.

The derivative of the convolution has a nice property, deriving wrt $f$ or $g$ is the same, but when one is an image and another is a filter, one can take the derivative of a filter that is a much better and fast thing! So differentiate wrt the filters.

### Autoregressive process

A time series (autoregressive process) AR of order K is the linear system: 

$$
x_t=\sum_{k=1}^K a_k x_{t-k}+\epsilon_t
$$

This is the classic regression but for time series.
We're working with the signal itself to compute future signal. Approximating it...
$\alpha_k$ are linear coefficient and $\epsilon$ sequence of *Identically and Idependantly Distributed* values with mean 0 and fixed variance.

I've a vector $\alpha$ that contains the $k$ coefficients the error will be computed in a more refined way because isn't really true that all is i.i.d, error made at previous time steps matters on the successive time steps. Then we weight the errors to get better estimation.

$$
\epsilon_t + \sum_{q=1}^Q \beta_q \epsilon_{t-q}
$$

Model used here is ***ARMA***, Autoregressive Moving Average. Our parmeters we want to learn are $\alpha$ and $\beta$. This is linear model so that is the limit. You need as well the autoregressor $K$ (and $Q$), estimation of $\alpha$ is performed by Levison-Durbin recursion, matlab code is:

```Matlab
a = levinson(x, K); 
```

Looking at the error the least the error the most complex is the model, the complexity matters and $K$ represent how complex this model will be. Order is estimated via Bayesian model selection criteria. 
Let's call $\alpha^i$ set of autoregressive parameters fitter to a specific timeseries (dataset) $x^i$. Our data can contain different lenght time series, and after we get the set of parameter we obtain a transformation from irregular time series to a representation with same lenght.

We can do timeseries clustering, novelty detection and we can even encode the time in a lower dimensional space, so we encode the time series as a vector (flatten) and train with an MLP. It's an horrorific baseline, but it can be done...

## Spectral Analysis

Another way to anlyze signal is to view them in frequencies domain when working with sound and music for example see the [[First Midterm]].
Key idea: decompose a time series into a linear combination of sinusoids. (and cosines) with random and uncorrelated coefficient, basically fourier analysis.

- Time domain: regression on past values of time series
- Frequency: regression on sinusoids

### Fourier Transform

Discrete Fourier Transform (DFT) is a linear transformation that maps a time series to a frequency series.  Can be easily inverted (inverse DFT) if you work in frequency domain is easy to handle periodicity of time series.

For fourier lovers: This is an oversemplification.

Given an orthonormal system ${e_1, e_2,...}$ for $E$ we can represent any function $f \in E$ as linear combination

$$ 
f=\sum_{k=1}<f,e_k>e_k
$$

With this orthonormal system:

$$
{\frac{1}{\sqrt{2}}, sin(x), cos(x), sin(2x), cos(2x),...}
$$

The linear combination becomes the ***Fourier Series***.
$$
\frac{a_0}{2}+\sum_{k=1}^{\infty}a_k cos(kx)+b_k sin(kx)
$$
Where the $a_{k}, b_k$ are coefficient obtained by integrating $f$ with the $\cos$ and $\sin$.

We can as well go into the world of **Complex values space** to represent our function in a circle (remind the example from 3b1b).
![[Pasted image 20230226185159.png]]
![[Pasted image 20230226183942.png]]

$$
\hat{f}_k=\sum_{t=1}^N f(t)e^{-i2\pi kt/N}
$$
  
Consider a discrete time series $x_t$ with $N$ samples. Using exponetiation formulation we get ${e_0, e_1,...,e_{N-1}}$ vectors $e_k \in C^N$.

These are the root of unity, we want to compute $c_k$ that is the integral (in discrete case is summation) of the elements in the orthonormal basis, so adding up every element.

$$
x_k=\sum_{t=1}^N x_n e^{-i2\pi nk/N}
$$

Given a time series $x = x_{0}, x_{1}, x_{2}... x_n$ it's **Discrete Fourier Transform** is the sequence in frequency domain, as we can see *above*.

The following is a visual representation on how our function $g(t)$ can be winded on the unit sphere and what we're trying to do is evaluating the mean of all the sampling on this function, of course when the frequency is the same of the fucntion this will become centered in $1$ with the *"center of mass"*
![[Pasted image 20230226184113.png]] Credits to 3Blue1Brown.

This is also the spectral representation of the signal.  So representing signal in frequency domain. The **inverse** transform is

$$
x_n=\frac{1}{N}\sum_{k=1}^N x_k e^{i2\pi nk/N}
$$

Be careful now we get the original time series based on the frequency. Complex conjugate.

### Basic Spectal Quantities in DFT 

We would like to measure relevance/strenght or contribution of target frequency bin k.

*Amplitued* and *Power*.

Power spectrum plot : we can plot the power of a signal, that is a distribution and it basically says where your signal distribute in frequencies, maybe you don't want some frequences and you can get rid of what you don't want.

Power spectrum image

DFT in action, but in spectral domain to train a predictor, use the $X_1,...,X_K$ as representing signal. Represent in spectral domain can reveal patterns that are not clear in time domain.

%image seen at lesson

%Closing offices at nigth image  


### Spectral Estimation

We have a lot of spectral analysis tools; for example *Spectral Centroid*, *Spectral Weighted Average* frequency between bands $b_1$ and $b_2$.

This is intresting to understand when you have music signals, for example what instrument are playing, centroids gives you what instrument is playing in a certain spectrum, this is nice for genre classification, also you can use it for RNNs; if you fourier decompose the behaviour of recurrent neuron (layer) in RNNs you can see the spectral centroid moves, that means that the network analize different things in each different layer. You can understand if you RNNs is behaviouring as a filter of a certain type. 

### Higher order moments

- ***Spread*** (variance-like)
- ***Kurtosis*** (4^th order moment) Measure flatness or non-Gaussianity of spectrum around centroid

For example you fourier decompose and then you go to see the Kurtosis, where has a peak you've a non-Gaussianity, you can use this to detect anomalies in time series.
Also Spectral Entropy if you plot this for different signal you can do pattern match of sounds representing the "peakness" of the signal.

### Take home lesson

```ad-summary

Old school pattern recognition on timeseries is still useful most of the time are linear methods, and fourier analysis allow us to identify key frequencies in the signal and these are powerful, but we can apply them to all sort of thing that have a frequence.

Example if you have a graph in fourier domain by represeting it in adiacency matrix and then apply that to get an orthornomal basis so you can use fourier on Graphs!
```

---

# References




