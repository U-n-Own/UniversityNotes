[[Statistics]], [[Probability]], [[Machine Learning]]

This method was introduced in the paper [Bootstrap Methods: another look at the jackknife](https://www.jstor.org/stable/2958830)

Is a technique used to estimate a **[[Sampling Distribution]]**.

```ad-abstract
What if we can *mimic* new dataset by seeing our small dataset to enhance data?
```

Like a *Statistical Inception*...

![[Pasted image 20231027142824.png]]

So it is a *Resampling*!

## Theory behind Bootstrapping

Theory is hard but this is a very simple idea at the end, randomness in data is randomness in estimator.

Let's call $T=f(X)$ our estimator where $X = (X_1,X_2,...,X_n)$ data distribution. The structure of $T,X$ is described as a PDF $f_T(t)$, just describing the likelihood of seeing different values of our variable, and the CDF $F_T(t)$ as how much probability is "stored" behind different values.

Both these two describe the distribution possible values but in different fashion.

Let's say the goal is estiamte the distribution of the sample median, so we want CDF. We can get from it sample median, sample mean and variance, but we can also get them instead as a *function of the data* as a *function of the CDF*. This will be a function that takes a function (a functional), of course we cannot know the real CDF but we can estimate it very near, how do we do?

Bootstrap says that we can estimate CDF: **Bootstrap Estimator**(or distribution) : $\hat{F}_n(t)$. We need data the $n$ is the data avaiability.

	ps: The hat symble is used to denote estimators in statistics.

$$
 \hat{F}_n(t)) \Rightarrow^{Close} F_n(t)
$$

We want to be close to the sampling distribution. Example is the *law of large numbers*, but when two function are close? How do we quantify a distance between functions?

## Uniform norm

Let's call $\sup_t |f(t) - g(t)|$ the "supremum" over all the possible values of these fucntions can take, if the widest difference go to zero, the fucntions will be very similar

This thing is called **[[Uniform Convergence]]**.

Now comes the hard part

1. Both fucntion change shape depending on the sample size
2. The bootstrap estimator is infeasible when sampling with replacement we get a $n^n$ growth

We use a subsample of these data that will be called **[[Monte Carlo Estimator]]**: $\overline{F}(t)$. 

So we go from MC estimate to Bootstrap to the one we need to approximate.

For MC we don't need a lot of bootstrap dataset (just 100 or 1000) to converge (uniform convergence here). As long number of bootstrap data is large, this take away problem 2, to estimate the bootstrap with MC.


$$
 \overline{F}(t)) \Rightarrow^{\text{Needs large B}} \hat{F}_n(t) \Rightarrow F_n(t)
$$

How do we solve the changing of function by change of samples?

Denoting limiting distibution $n \rightarrow \infty: L$ and $\hat{L}$ for the estimator
we need a final link within these two, usally we can parameterize these two family, so if the estimator variance and true variance are near each other also the limiting distribution will get close. 

Thing is: Bootstrap variance will approximate the population variance with large enough sample size.

Condensating all: We need a large number of $B$ by using MC to converge to the estimator bootstrap and then we need the sample to be so large to approximate limiting distribution so large $n$.