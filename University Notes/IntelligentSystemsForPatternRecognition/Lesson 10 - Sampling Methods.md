Date: @today

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Approximating Expectations

What is sampling? Properties of intrest, why is relevant in this setting, and from what distribution we want to sample, and some general classes of samples. Sampling methods used in practice with Botlzmann machines

## What is sampling

Realization of one RV of a model, so a set of realization, classical example is to sample dice process.
![[Pasted image 20230321111419.png]]
What i can build is a list of samples with $l$ as index and their outcomes $x^l$. 
What we can do with sample? Approximate expecations! 
Because if the distribution is untractable and we want to estimate the $\mathbb{E}_{p(x)}[f(x)]$.

We cannot enumerate all states of $x$. But we can compute that!

$$
\mathbb{E}_{p(x)} \approx \frac{1}{L}\sum_{l=1}^L\limits f(x^{l}) = \hat{f}_x
$$

There are many intractable cases: distirbution of a **Boltzmann machine** or the posterior in LDA as we saw in previous lesson [[Lesson 9 - Bayesian Learning and Variational Inference#Latent Dirchilet Allocation]]

## Why need to sample
Parameters from models comes from distributions, there is some $\alpha'$ that is from what we pull our $\beta$s
Even the parameter of the multinomial are in the left side of posterior, so we don't know them.

We can extract $\beta$ by drawing samples from distribution.

```ad-important
`Sometimes sampling becomes actually our learning`
```

So inferring information from $w$*

![[Pasted image 20230321112004.png]]

### Properties of sampling

We know that empirical distribution converge to the true distribution.
![[Pasted image 20230321112408.png]]

By certain times we're not even drawing samples from that distribution but from a proxy of that, not even the real, this make even more difficult to approximate it.

- Sampling appoximation  $\hat{f}_x$ of the expectation can be an *Unbiased estimator*. Drawing from a different distribution but that has a property that tells us that we're not that far from the real one. **Valid approximator**
- Sampling approximation $\hat{f}_x$ of the expectation can have *low variance*. So you can have few samples that approximate quite good the expectation. If there is high variance then**How much pain you're gonna pay to get good approximations**


We want to have both properties

## Unbiased estimators

This ensure that we're sampling from a valid distribution. This is the base of what we want, this is the easier thing to obtain.
![[Pasted image 20230321112802.png]]

## Variance of Sampling approximator

This tells us how much trust we can give to the samples, if they are very far away from each other, to get a good approximation we have to draw a big number of samples. For low variance what we get is always close to the true distribution.

![[Pasted image 20230321113003.png]]

How do we ensure this?

Given a samples we want that they are indepedent and have the same marginal. So given this we can sample a small number, but what is this is not the case?
![[Pasted image 20230321113144.png]]

## Sampling in practice

There is a tradeoff, some sampling procedure are more costly than other, some are more efficient but sacrifice independence between samples. Again no free lunch theorem, pay less but have to sample more because you've high variance.

The $\tilde p(X)$ can be a very big joint distribution, so we need to decide what variable sample first, then we start to do approximation, and changing distribution from which we draw the $p(x)$

So $\tilde p$ is the one that depends on the sampling procedure: properties of sampling procedure depends on $\tilde p(X)$. And we want a valid sample (unbiased estimator) and low variance as properties.
![[Pasted image 20230321113553.png]]

### Univariate sampling

Drawing samples from univariate distribution so discrete RV, and there is a single RV. What i need is only to have a random numbe generator that spits number from $[0,1]$.
I pick a "stick", decide how much outcomes, 3 in this case, and assign a certain probability. If we sample a certain number from this univariate distribution we assign the number that is in the interval. If ends in red then is 1, if blue in the 2 and green in the 3. Then we estimate the expectations.

For the case of a continuous RV, we take the integral over $p(x)dx$, from a certain interval.

![[Pasted image 20230321114017.png]]

### Multivariate sampling

In this case we can have a lot of variables, a joint distribution of these with a set of five RV, our samples $x^l$ are assignment to all the values for each random variable, how can we sample from this distribution?

You create a $Sup$ random variable, that is a univariate RV, that is 1 when all are 1,1,1,1,1. This grows in a very fast way as number of possible outcomes and number of RV

![[Pasted image 20230321114334.png]]
This is equivalent space, but grows too fast.
![[Pasted image 20230321114611.png]]
![[Pasted image 20230321114646.png]]

Do we have a way to transform joint distribution in one single distribution?
![[Pasted image 20230321114727.png]]
Then we sample in order the $p(s_1)$ we plug it in the second sample and so on using the conditioning on very many observed at previous iterations. This is ok because we're drawing from a univariate distribution.
This becomes big, unless we can simplify those how do we do? **Markov assumption**! [[Lesson 7 - Hidden Markov Model#Markov Chain and HMM]]

### Ancestral sampling
I can sample in parallel in this way. First sample ancestors and then the others
![[Pasted image 20230321115053.png]]

I can reason on conditional independance assumption, this is quite nice.
![[Pasted image 20230321115233.png]]

### Sampling with evidence

What is the problem? Sampling with some evidence break things.

Every time i do sampling on evidential variable i have to see if these are part of a $V-structures$. Is as difficult as run complete inferences.
![[Pasted image 20230321115507.png]]
Run AS on old sturcture and discard subsamples is not good even.

## New sampling procedures
![[Pasted image 20230321121343.png]]

### Gibbs Sampling

The initial sample $s_1$ is generated and then the following  samples are generated picking one of the RV of $s_1$ and we get the sample $s_2$, so randomly picking RV and changing one at a time. This is a very fast way to get them is easy, but are dependant from each other and there is high variance, so we have to get many of them.
![[Pasted image 20230321121402.png]]

If i'm in a Bayesian network i don't care about the other that aren't in the markov blanket. So we can build a samples that select a RV $s_j$ and sample by sampling on the distribution we can see above by computing only the markov blanket. Dealing with evidence here is also easy
![[Pasted image 20230321121725.png]]

### LDA Gibbs sampling
We need to provide initial values for model parameter, assignment of topic to the words and so on, at each iteration we pick a random variable we change it $z_{ij}$ assignment of a specific topic to a specific word is drawn from the previous distribution. Then you can use the new $z$ to approximate the new $\theta$ and then the same for $\beta$. Is not that different from EM, istead of compute expectation we compute samples, but we can get as good as we like just iterating more.

We're generating incrementally $\beta ,\theta$ trought sampling and actually do learning by sampling.

In order:
- $z_{ij}$ topic assignment for the j-th word in i-th document : Which topic the word belongs to
- $\theta{i}$ topic distribution for the i-th document following dirchilet distribution
- $\beta{ij}$ is the word distribution word the j-th word in the i-th topic 
![[Pasted image 20230321121958.png]]

To get a valid sampling procedure we want to be careful, so we want to get the $q$ right, in a form that will get a good estimate. 
Here we want to ensure $q$ as stationary distribution at infinity and that stationary when $t$ is infinite $Q(x^t|x^{t-1})$. So those markov chains have to converge to our posterior we're sampling from.
What we're hoping is to get a valid sampler at the limit.
We're drawing samples form a markov chain, they're not independant, how do we make them independant or more independant in our possibility? How much information a certain state can store, is limited obviously, but if we have a lot of nodes between two nodes the information will be very far away.
![[Pasted image 20230321122540.png]]

## Markov chain Monte Carlo methods
Sampling on our $q$ in a smart way, we get different type of MC algorithms.
![[Pasted image 20230321123534.png]]

![[Pasted image 20230321123721.png]]

### Take home lesson

We discussed sampling at high level is a tool that is useful for intractable distribution, usually **posteriors**, the nasty guys. They are conditioned from observable variables, ancestral sampling are valid but if we introduce observation all falls, and becomes a complex problem, so we use approximated algorithm, creating good $q$ distribution with some good properties, in all cases we try to approximate expectations. The empirical expectation comes from $q$ by drawing samples.

Tomorrow: ***Boltzmann machines*** the missing link between MRF and RNNs. They have a double interpretation of these Boltzmann machines, and Restricted Boltzmann Machines with Autoencoder that introduced the *Deep Learning stuff*
```ad-summary


```


---
# References

