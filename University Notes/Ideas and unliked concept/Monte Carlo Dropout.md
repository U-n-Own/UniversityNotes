[[Machine Learning]], [[Regularization]], [[Dropout]]

From:
[Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf)
## Why introducing MC?

Usually *Dropout* is a technique used in training to get a better average over an esamble of possible models to acquire better generalization capabilities, MC methods adds a repeated sampling to approximate a distribution as we know from [[Lesson 10 - Sampling Methods]].
## MC Dropout - the catch

This method can be intrepreted as a [[Bayesian Approximation]] of a [[Gaussian Process]].
We can treat many different networks (ensamble) as our Monte Carlo samples! So we're sampling from the space of all possible models.

How does it work? 

```ad-important
We simply apply dropout at test time, that's all! Then, instead of one prediction, we get many, one by each model. We can then average them or analyze their distributions.
```

And the best part: it does not require any changes in the model’s architecture. We can even use this trick on a model that has already been trained! To see it working in practice, let’s train a simple network to recognize digits from the MNIST dataset.
### Why reducing uncertainty

Neural nets for regression and classification do not capture uncertainty, we saw that for that [[Bayesian Networks]] or in general where we want to get a certain *posterior distribution*: $p(\theta|X)$
We encoutered these models here [[Lesson 5 - Generative and Graphical Models]]

Why would we do this? In order to reduce the uncertainty of our model in prediction, usually Bayesian methods comes with a big problem, true posterior is unfeasible to obtain (growth is $2^n$ where $n$ is the number of RV).

In the paper they explain how a dropout layer after each weights layer equivalent to a probabilistic **deep [[Gaussian Process]]**, marginalised over it's covariance function parameters.

In fact they showed how dropout objective minimises the [[Lesson 9 - Bayesian Learning and Variational Inference#Kullback-Leibler (KL) Divergence]] between the approximate distirbution and the posteerior of the gaussian process
