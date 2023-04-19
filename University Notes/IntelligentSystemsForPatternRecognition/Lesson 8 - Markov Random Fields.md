Date: [[2023-03-14]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# MFRs

Undirected graph $\mathcal{G} = {\mathcal{V}, E}$ : ***Markov Networks***

![[Pasted image 20230314113810.png]]

That factorization of joined distribution of RV, with C indexing the maximal clique, if a maximal clique is made of n nodes the potential function $\psi_C$ identifies. The term energy, potential (local potential, local to the clique!) pops out a lot because of physics related stuff.

How do we choose potential function? This need to be defined over the clique, to define a proper factorization

![[Pasted image 20230314114125.png]]

What are the low energy configurations?
We're choosign an exponential over a low energy function, $E(X)_c$ are derived from latent variables.

## Factorizing Potential Functions

![[Pasted image 20230314114602.png]]

How we build potential functions? We sum over the cliques, the learning parameter $\theta$ , where the $k$ are features, $f_{Ck}$ is a feature function that tells you if you're ok with the instantiation with that feature, maybe you have a clique with $X_{1,}X_{20}, X_{23}$ and that function tells you how you "like" that configuration.

For example if the $X_1$ is the color blue of a pixel, then if you like blue and red pixels, you can have different $f_{Ck}$ that tells you.

One can relax removing the $C$ from $\theta$ or from $f$. What happens?

Seeing up we're factorizing, because of the exponential of the sum. How do we factorize?

### Factor Graphs

Introduce this graph to make explicit the factorization of the clique. 
We have multiple way of factorize these using a certain $f$, we're introducing a new class of nodes that are the **feature functions**.

In the first case we have 3 edges having the factorization of the three, and then at right we have two nodes, the same as before and another $f_b$ a feature function that factorize two of them, and we can express more factorization.

We can as well have $f({x_1}) \approx P(x_1)$ so our *prior*. Is basically the feature function of a single node. Recall the thing we said on graphical models in [[Lesson 6 - Conditional Independence and Causality]]. Some nodes have no parents...

![[Pasted image 20230314115551.png]]

## Sum product inference

Class of **exact** inference algorithms (belief propagation). Using factor graph representation to provide a unique algorithm for our models.

Recall the markov random chain we had something like node and the note after in between we have a featur function: $P(S_{t}| S_{t-1})$

Inference is feasible for chain and tree structures, in HMM we have Forward backward, in markov random fields is computationally more expensive because we've not probabilities here, we've got only functions and most importantly: we can have dense connection because of the undirected nature of these models.

### Inference in general MRF

Restructure to obtain a better structure using spanning tree or Chow-Liu (approximation)
Or approximated inference (variational, sampling). In general what we do is:
![[Pasted image 20230314121211.png]]

## Restricting to Conditional Probabilities

If i have $P(X,Y)$ can i evaluate $P(Y | X)$ knowing that $X$ is known by conditional independance we can say $P_\theta(Y | X)P_\theta(X) = P_\theta(X,Y)$.

So we focus on learning that. That's what conditional models do. These discriminate a certain part, do not model that RVs and working only on what we don't know using what we know.
![[Pasted image 20230314122351.png]]
We just sum on the all unobserved random variable those we can see are the input data.
Marginalizing on Z, one sums over y, and obtain that values by marginalization

## Conditional Random Fields

We're modeling the white nodes given the one in blue (observable ones).
We've got a clique there with those three functions, the product of those three distributions give the potential function of the link
![[Pasted image 20230314122538.png]]

## Feature functions

What feature function do?

This feature function tells us how muhc we like a specific oberved pixel to be in a certain cluster, then the connection to $y_{i,}y_{j}$ gives you contraints, usually nearby pixels are in the same class.
![[Pasted image 20230314122718.png]]

## Discriminative Learning
 
 Given $Y, X$ with the latter observed ones, let's simplify with single $Y$ and multiple $X$, then we can observe $Y^n$ corresponding to $X^n$ for some samples.

### Logistic regression is simple conditional MRF

We can see that

$P(Y | X) = e^{\sum_{i=1}^N F(X_i,Y_i-1, Y_i, i)}$



## HMM Vs. CRF

In CRF we give a general description of what we'd like to see for example for language, or NLP, we can say tha something is a name if starts with a capital letter, so for example $x_{t}$ is th word and $f_e$ gives that information.

![[Pasted image 20230314123845.png]]

We can express relationships between past and futures "tokens" (in NLP). This enhance our model.
![[Pasted image 20230314124036.png]]

Product of cliques over feature function, because feature function doesn't depends by time, neat!

## Generic LCRF formulation

We're just making dependant the state $y_t$ on the $f_i$ at the previous time step.
the $\mathbb{1}$ are indicator variable that are 1 in a spot and 0 in all the other. Like dictionaries described in a math way.
![[Pasted image 20230314124424.png]]

## Posterior inference in LCRF

Equivalent for ***smoothing problem***? $P(Y_{t},Y_{t-1}| X)$ solved by exact forward backward inference

- Sum product message passing on the LCRF factor graph
![[Pasted image 20230314125004.png]]

We bring the cliques ***local*** potential functions when we forward the message from the past and when we backward it from the future.

## Other inference problem

Solving true complex problem is rather impractical for real graphs.

Classical setting on finding our $\theta$ we are doing max conditional log-likelyhood
![[Pasted image 20230315161937.png]]
Supervised setting the conditional is the *dataset* made this way $D = \{ (x^1,y^{1}),...(x^{n},y^{n})\}$

Where $n,t,k$ are the sum over $n$ samples, all the time in the sequences $t$ and the feature function $k$, that maybe capture different relationships between one and the next, like HMM

To solve this problem we have to optimize by computing the derivative wrt $\theta$. Fitting discriminative models something is **Regularized** otherwise the model will learn by heart and not generalize, so we need to punish the model, this is the term that is ***regularizer***. How do we get this regularizer? (This is a classical L2 penalization, to get a different one for example we assume a poisson and we get a L1 penalization)

![[Pasted image 20230315162446.png]]

We introduce some prior, the $P(\theta)$. These are iid. So if we apply a log to this we get

- Deriving the L2 penalization
$$
P(\theta) = -exp(\frac{\theta^{2}_{k}}{2\sigma^2})
$$

Our $Z(X^n)$

$$
Z(X^{n)}= \sum_{t}\sum_{y,y'} exp(\sum\limits_{k}\theta_kf_k(y_t,y_{t-1}, x^{n}))
$$

Then we proceed to evaluate

$$
\partial \frac{Z(X^n)}{\partial \theta_{k}}=\frac{1}{Z(X^{n)}}\sum\limits_t\sum\limits_{y,y'}exp(\sum\limits_{k}f_k(...))
$$


$\mathbb{E}[f_k]$ is our empirical expectation over our data, so we pick our y,y' and we plug them in to get this expectation. For the posterior we do not compute entirely on the data, because we're marginalizing our the y, so if we want to find a solution, the optimal will be when the expectation under model posterior matches the expectation over the data, this is when we get the optimum. This is a rather intresting property we need to know. 

```ad-quote
"Hey model what do you think about this x, what do you believe? Oh yeah they're great!! Well i'm going to look at the data, hey data what do you think? ... EHhhh... [Iterate this way]"
```


## Stocasthic Gradient Descent

Then we have our $\sigma$ that is an hyperparameter that regulate how the models is `rigid`
![[Pasted image 20230315164513.png]]

## Engineering features

This models: (Markov Random Fields) are very good for structured prediction, because these prediction isn't a flat output, it's more like an output of hierarchical prediction, so you predict a superclass with subclasses and so on (tree like). 
If you know that there is a lot of classes and between them there are a lot of common things then.
For complicated output spaces this become intresting. 
![[Pasted image 20230315164651.png]]

## Limits of CNN

We cannot help a CNN to a understands that the sky is up or other things like those, but adding MRF, we can do a pre-filtering before applying the CNN to get much better results.
![[Pasted image 20230315165100.png]]

![[Pasted image 20230315165405.png]]

![[Pasted image 20230315165420.png]]

## Hierarchical semantic segmentation

With MRF, we can extract the level of the images, reasoning on single pixel, then reasoning on some more complex pattern, putting togheter information and then infer a lebel for the objects so we can infer what's about the image, and describe it as well! For example every time we see a line (black) that is a feature function, that describes part of an image the $z_k$ are feature function that says how probable is to see a person given the image and the piece of it.

![[Pasted image 20230315165553.png]]


Another example

For example if in the image we have some stuff that says that there is Sea, then other that says Car, then the probabilities to car are lowered, is very unlikely to have a car in the sea.
![[Pasted image 20230315165846.png]]

Alignment of information: We collect information by pixel of different camera and then we put on top another CRF to factor the information about all the pixel in the different cameras and how they interact with them.
![[Pasted image 20230315170116.png]]


### Take home lesson



```ad-summary


```


---
# References

