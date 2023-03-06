Date: [[2023-03-02]]

Status: #notes

Tags: [[IntelligentSystemsForPatternRecognition]], [[Probability]]

# Graphical Models or Generative

Generative learning is when we have parameters of a certain probability distribution, we can generate from a model also if we approximate enough.
If we have large avaiability of labeled data a discriminative model will work better, instead generative even if you have class information is used to *model a distribution of the data*, and when you do it with enough accuracy you could generate samples. Since NNs are universal approximator they do not assume any family of distribution for generation, instead here the assumption a priori are important! So reasoning on prior knowledge is something very intresting, giving a model a prior knowledge ins't that easy. If one want to model some task needs big models and we need to... approximate to break down complexity.

## Graphical models framework

We do graph based task, that usually are $NP-Complete$.

For machine learning usually *Inference* is the prediction, but in general is a concept that describes a process to infer some knowledge to *learn*, so also learning is inference.

Let's define formally: 
```ad-important
A graph whose **nodes** are random variables whose **edges** (links) represent probabilistic relationships between the variable
```

So depending on the edges we have different models!
![[Pasted image 20230302150022.png]]

In the first we have the nodes, and we can say that the directed node is entailed by the others.
No cycles here, because casuality can't have them!

The undirected graphs are good if we want that things obey some costraints

The dynamic models are whose in which random variables can appear in time and disappear. Use them for dynamic systems: **Petri nets**?

Example we want to give each pixel on an image a certain probability to be in a class
![[Pasted image 20230302150553.png]]

An example of prior is that all from a certain high of image will be likely the sky or mountain and the part below will be terrain.

### Variational Autoencoders
![[Pasted image 20230302150748.png]]

## Refresh on probability

A random variable measure the likelyhood of something discrete or continuous a RV models an attribute of our data $P(X=x)$, in machine learning all is a RV, also features: age, gender, profession (example). There are RV for which we don't have examples, inputs units have an output (the input itself), for the hidden neurons we don't have examples, they're not observable.

### Types of RV

- Discrete: $P(X = x) \in [0,1]$
- Continuous: A density function $p(t)$ and define a density distribution.

## Joint and Conditional Probabilities

We have probability distribution of random variables, for example factorization can be for independant variables, one can multiply the marginals to get the joint probability.

**Conditional probability**
$$P(x_{1},x_2,...,x_{n}| y)$$

$y$ is a prior knowledge that gives us information, for example forecast for prediction of raining, when you write a probability what are you really doing is getting a family of distributions if $y \in \{1...C\}$, so for two random variables without conditional, we get a matrix, adding a conditional we get a tensor in which the $y$ will be our slice or depth of the cube for example. Of coure we can generalize this to more RV.

### Chain Rule
Why use chains rule? In hope to simplify the things.
![[Pasted image 20230302153330.png]]

Hidden variables are introduced in models by marginalization, this because learning a models is basically learning that distribution. **Latent variables** 

### Bayes Rule (ML interpretation)

Given an hypotesis $h_i \in H$ and observation $d$

$$
\color{yellow}{P(h_{i}| d)} \color{white} = \frac{\color{red}{P(d|h_i)}\color{blue}P(h_i)}{\color{green}P(d)}
$$


The $\color{red}{first \; term}$ in the second part is the *likelyhood* so how much probable is our data  so probability of our model to generate observed data, the $\color{yellow}{yellow \; term}$ at right is called *posterior* and the second $\color{blue}{term}$ at numerator is the *prior*, that gives us how good is the model. The $\color{green}{denominator}$ is the *marginal likelyhood*  but is also called *evidence*.

A practical application of marginalization over all possible hypotesis is to rewite bayes like this
![[Pasted image 20230302154036.png]]


### Independance and Conditional Independance

Two RV, $X,Y$ are independant if 

**Conditional** : If we know things about a third variable
eg: I pick up umbrella : X, i pick up sunglasses: Y, then i've an observation : it's raining.
Knowing that is raining make them independant, instead before they weren't! Thing become independant if we know the right things. 
![[Pasted image 20230302154217.png]]

## Inference and Learning probabilistic models

**Inference** how can we determine the distribution of one or several RV given the observed ones?

Usually we have an hypotesis $h_\theta$. How do we do inference? How do we infer the values of $X$ given $d$?
![[Pasted image 20230302154851.png]]

The first after the sum is the prediction given the hypotesis, term number 2 is weighting the prediction given the posterior, then what we do the better the model is in according to posterior the way we fix the weightning will decrease... 
This is called Bayesian Prediction, the is a theorem about this that says that nothing can beat a Bayesian predictor. But we cannot we go trough all the possible instantiation, this is not feasible, so we pick up the best hypotesis, but what is the best? The one that has the better posterior, the **Maximum a Posteriori Hypotesis** (we pick the best $P(h_i|d)$) and then we take a decision based only on that.

Thanks to the Bayes rule we can get $P(d|h)$ if we know the other one, the best we picked before. If all hypoteses are a priori all the same probability we're intrested in the **Maximum likelyhood hypotesis** so rather than the posterior we want to maximize the likelyhood. This is the maximum likelyhood estimator. Usually parameters are obtained by ML estimate. The presence of the prior influences what we use, because it's effect will be useful for Maximum a posteriori, but the prior is good only if we have few data, otherwise our prior on that collection of samples will be much less important. 

![[Pasted image 20230302155319.png]]

This works also for fitting a model!
Maximizing the likelyhood of our data, so how much the model is good at explaining data, in fact MSE is a proxy for likelyhood, but also take in consideration prior knowledge, maybe we penalize model that are too complex, this a measure of *regularization*. The sum of squares of $\theta$... finish here

#### Inference and Learning in Probabilistic models

Recalling last lesson.
**Inference**: How can determine distribution of values of one or more RV given the observed of others? 

$$
p(graduate | exam_1,...exam_n)
$$

##### Apporaches to inference

Bayesian consider all hypoteses weighted by probabilities, we rely on what we know, in a discrete word take these hypotheses by marginalization.

Our optimzation problem will be described in this way

$$
\theta_{ML} = arg\; max_{\theta \in \omega} P(d|\theta)
$$

The prior in ML are made in a way such that will punish classifiers with high parameters, it acts like penalization term, writing an appropriate $P(h)$ prior hypoteses embodies something like

$$
Regularization := -\lambda\sum\limits_{i}{\theta_i^2}
$$

## Regularization

Thanks to information theory we can exploit log in this way
![[Pasted image 20230303143436.png]]
Minimum description principles is a thing that try to minimize the bits to represent informations
For zero bits we just have a lookup table.

#### MAP vs ML

Maximum Likelyhood (ML) and MAP are *point estimates* of Bayesian, considers a space in which $\theta$ veries, when you take a Bayesian approach you sum over all possible values of the $\theta$ space
and use a particular instance of the points. (Integrating usually).
In MAP you're just picking up the maximum.

## Maximizing Likelyhood
![[Pasted image 20230303143810.png]]
Problem comes when you have **hidden variables**, variables that we have no data to plug and see how they are done. We have a set of observed random variables $x$ (training data), and unobserved hidden or latente variables $z$ (data clusters). How to solve?
![[Pasted image 20230303144017.png]]
We try to maximize the *complete likelihood*, where we have observable variables and unobserved, we predend to have observed then and we do *chain rule* to decompose them.
Since $z$ are not avaiable we introduce them by **marginalization**! 
We can't do it in a *closed form*, here's why we have the $\theta^{k+1}$, we start with random initialization of our parameters, these are the parameters are time 1 and you feed and go on until your likelihood flattens then you stop. There is a theorem that says that expectation maximization algorithm yields a non-decreasing likelyhood, flattening is the worse case. 
You don't need to get the $z$ right, you just want their expectation. We're not maximizing the completely likelyhood we're just maximizing in expectation wrt $z$ this doesn't give us the same solution of the exact problem but gives us at least a local optima.
If we can't compute posterior we can push the lower bound (Variational expectation maximization we'll see later), we will use an optimization function called $Q$ (basically this is a Neural network.)

## More on Graphical models

If you have a joint distribution of $N$ RV, the space to compute them (simple RV) will grow with $2^n$, we take assumption to break these joint distribution (what we want).
Graphical models are used in these cases, we can cast our knowledge and distribution in a graphs and apply then inference, if the graph is directed we get *Bayesian networks* and for undirected we spoke about *Markov random fields*.


### Take home lesson

```ad-summary


```


---
# References

First chapter of Barber books cover the refresher on probability
Generative models in code: Tensorflow probability, PyMC3, Edward, *Pyro* 

