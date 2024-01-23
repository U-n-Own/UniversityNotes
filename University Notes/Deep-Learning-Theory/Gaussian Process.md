[[Machine Learning]], [[Probability]], [[Bayesian Learning]]

# Infer a distribution!

![[Pasted image 20231130161157.png]]

## Bayesian Linear Regression 

Simple regression that fit a line and estimate noise and variance, but this is missing an important factor : **Uncertainty**!

![[Pasted image 20231130161548.png]]

How do we estimate? Well it's the title.

Let's say that a *Model* produces sample of functions prior to the obervation of data

These are the type of samples that are produced by our model:

![[Pasted image 20231130161645.png]]

So : *Prior Samples (Model)* + *Data* $\rightarrow^{\text{Bayes Theorem}}$ *Posterior Samples*

### Sampling

For example these would be a posterior samples of these data, then we could apply a noise-variance to each sample (the lines).

![[Pasted image 20231130161958.png]]
### Averaging

 And then average over a big number of sample $n$ (for $n \to \infty$ we get precisely the posterior distribution)
 
![[Pasted image 20231130162140.png]]

# GP generalize Bayesian linear regression!

What we do for gaussian process is just a generalization of Bayesian Linear Regression, in fact our sample are over general function (smooth function)

![[Pasted image 20231130163130.png]]

This is an example of predicting with gaussian process

![[Pasted image 20231130163148.png]]

## How to control GP?

Which functions are more likely to be sampled from a GP? Well it depends from the **[[Kernel]]**.

Now we depends on the kernel for sampling our function and we need to defined a good way to measure when two $x$ are similar. 

Most common Kernel in this case is the [[RBF Kernel]]

## Modeling with GP

Usually we have a moltitude of Kernel we can choose and typicall we have a set of kernel we could do operation like, adding kernel (adding kernel correspond to add sample functions), multipliying (multiplying sampled function do not corresponds to but it's a useful approximation ignoring that)

### Modelling example

Let's define two kernels

![[Pasted image 20231130173702.png]]



#### Linear Kernel

![[Pasted image 20231130173734.png]]
#### Periodic Kernel

![[Pasted image 20231130173756.png]]

### Let's fake some data

These seems like a quadratic fit, so let's try to multiply two linear kernels

![[Pasted image 20231130173818.png]]

#### Multipliying two linear kernels

Sample

![[Pasted image 20231130173928.png]]

Multiply

![[Pasted image 20231130173954.png]]

### Posterior predictive fit

Now if we sample an amout of these quadratic models some will fit the trend of quadratic growth in the data, but we need to add uncertainty to fit really good so we *add* the periodic kernel and sample from these new kernel

Well we faked the data and found the right hyperparameters so do not be proud this model is not what we find in real application

$$
\textbf{Model Kernel} = \mathcal{K}(x,x'|\tau) = (v_1xx')(v_2xx')+\exp(\frac{2}{\mathscr{l}^2})\sin^2(\frac{\pi}{p}|x-x'|)
$$

![[Pasted image 20231130180004.png]]

## Gaussian Process Assumption

We assume that observation and test points are *Jointly distributed* as an $(N+M)$-dimensional multivariate Normal:

In mathematical terms, let's say you have a collection of input points $X={x_1,x_2,…,x_n}$, and the corresponding function values $f={f(x_1),f(x_2),…,f(x_n)}$ from a random process. The Gaussian process assumption states that the joint distribution of the function values ff is a multivariate normal distribution, meaning any linear combination of these values will also be normally distributed.

Reformulating:

Let's denote the training data as $X = \{x_1, x_2, \ldots, x_N\}$ with corresponding function values $f = \{f(x_1), f(x_2), \ldots, f(x_N)\}$, and let $X_* = \{x_{*1}, x_{*2}, \ldots, x_{*M}\}$ be the set of test points. The joint distribution of the training and test function values is given by:

$\begin{bmatrix} f \\ f_* \end{bmatrix}$ $\sim$ $\mathcal{N},(\begin{bmatrix} \mu \\ \mu_* \end{bmatrix}$, $\begin{bmatrix} K & K_* \\ K_*^T & K_{**} \end{bmatrix}$)

Here,
- $(f)$ is the vector of function values at the training points,
- $(f_*)$ is the vector of function values at the test points,
- $(\mu)$ and ($\mu_*$) are the mean vectors for the training and test points, respectively,
- $(K)$ is the covariance matrix between the training points,
- $(K_*)$ is the covariance matrix between the training and test points,
- $(K_{**})$ is the covariance matrix between the test points.

The key idea is that this joint distribution is a multivariate normal distribution, and from this joint distribution, you can derive the conditional distribution of the test points given the training points, allowing you to make predictions with uncertainty estimates using the Gaussian process framework.

... finish this