Date: [[2023-03-15]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Generating distributions from distribution

Let's intorduce Variational Inference so relax the fact that i cannot compute the exact posterior, if we don't have the $\alpha \;and\; \beta$ terms. What if we can't compute those?

## Latent variable models

![[Pasted image 20230315171455.png]]

The assumption is that we have some observable data, we know that $Z$ server to explain $X$ (a very high dimensional collection of variables) but we don't know $Z$ and know $X$. When i observe $Z$ the vector $X$ components become conditional independants, generally they're not.

![[Pasted image 20230315171644.png]]

We introduce $Z$ by marginalization (the integral) and then we apply the factorization, the product can breakout $X$ in all his components.


## Latent Space

![[Pasted image 20230315171834.png]]

Where $N$ is the size of $X$. This pictures resembles something we've just seen! ***Autoencoders!***



## Tractability

We're introducing coupling between distribution, that can be depending one another, mutual dependancies won't allow to compute closed form posterior!
In the simplest case we cannot compute posterior, but in most difficult cases we can't even get the optimize likelyhood.

We want a version of EM to get solution without knowing the posterior!
![[Pasted image 20230315172601.png]]


## Kullback-Leibler (KL) Divergence

It measure how close two distribution are each others.
We want that our $p(z|x)$ in a form posterior like but we don't know $z$. The divergence is ok when both $q,p$ are high, so they are close to zero, if $q$ is high and posterior is low we're not approximiting good enough the posterior. If $q$ is low we don't care.
![[Pasted image 20230315172711.png]]

## Jensen inequality

If you linearly combine point and compute $f$ then you're lower bounding by some value: this is for convex
![[Pasted image 20230315173321.png]]

...

Finish taking note until the 13 slide



Idea we saw last time is to choose a $Q$ such that it can be computed.

We're appling the EM to the lower bound. $\mathcal{L}$ is the lower bound
![[Pasted image 20230316142759.png]]

## Generatie models for multinomial data

Bag of words assumption is a simplification that says: you count the words in a text. Without the causality relationships between the words.
![[Pasted image 20230316142921.png]]

We can see a representation for text, but this works for multinomial data as graphs or images.
![[Pasted image 20230316143113.png]]

Given a dataset in a matrix form we can represent images as well as we will see later.

Latent variables are unoberved random variables, so these latent variable cluster information, in particular take the name of **topic** used for documents for example. If we have some latent variable and a list of words that co occur, like "Ball", "Player", "Goal" etc... this will be represented by a certain latent variable for sports, in a completely unsupervised way. In one document we can have different part, speaking of different topics, so we want to model documents as a mixture of topics.

We're gonna to assign to each word a topic, for example color are topics in this matrix. Another fun things is that a word (row in matrix), is red and then blue at a certain point, this capture the polysemy of a word. We're modelling semantic over words and documents.

![[Pasted image 20230316143610.png]]

## Latent Dirchilet Allocation

If you remember this is a Bayesian network with replicated relation, this is saying, that all the RV are within M, replicated M times, so for each document we have these RV, then, within each document we have an assignment of $\theta$ and then we have another level of nesting, for each of $N$ words in the documents we have those parameters $z,w$. $\beta$ apply to each words (hyperparameter), and assign the number of possible topics or cluster for that word in our data. That $\beta$ will describe a matrix with column as topic and given a topic we have a probability for each words.

We're picking an high dimensional document of $N$ words, and we want to reproduce those with $z$.
![[Pasted image 20230316143811.png]]

But what is $\theta$? Well this is a global probability of all the document, so what is the `color` of the entire document? It's a multinomial distribution other than a parameter, we're generating by sampling on $\theta$ and that is derived from $P(\theta|\alpha)$ this is why is called Dirchilet.

$\theta$ is a vector that sums to 1 basically (multinomial), by setting the fact that $P(\theta|\alpha)$ as a Dirchilet(Is a distribution that return a distribution from a distribution)
One can have a subtree of distribution...

### Dirlichet distribution

![[Pasted image 20230316145221.png]]

If the distribution are conjugate then you can multipy your distributions.

### Geometric interpretation

Having as vertices $w_{1},w_{2}, w_{3}$ (obviously think this in higher dimensions)
Embedded here there is a lower dimensional representation and we an find our document in that part into the red triangle, and we can  use a new coordinate system, $z_{1},z_{2},z_{3}$, usually words are $50k$ while the topics are like $20,30$
![[Pasted image 20230316145552.png]]
LDA is putting a distribution over the space over the internal triangle.
![[Pasted image 20230316145840.png]]

So $\alpha$ control if the $x$ is near vertices (few or 1 topics), or near the center of this triangle then will have a lot of topics, so $\alpha$ tells us how we restrict the model is sparse.

### Effect on $\alpha$
Big alpha = 100
![[Pasted image 20230316150119.png]]

This (Dirchilet) is used in ML, because in deep learning you can use this $\alpha$ regularizer

### LDA and text analysis

![[Pasted image 20230316150627.png]]

### LDA generative process
![[Pasted image 20230316150911.png]]

In the last term we can use conditional independance for $P(\theta|\alpha)$ and the productory for each words that simplify our word.

The first probability is a Dirlichet and the other two are multinomials (those i multiply)

## Learning in LDA

How do we train these models?

If we see the likelihood of the documents we need to introduce the hidden variables, $z$ is introduced by maginalization but $\theta$ is continuous so we have an integral.
Ths is the likelyhood we want to maximize, of course we're maximizing the log likelihood. But since we've got an integral and we know that expectation maximization has not a closed form we're screwd!
![[Pasted image 20230316152323.png]]

The important thing is that we want to infer posterior, that is intreatable
![[Pasted image 20230316152616.png]]

Why is untratable? well the denominator is quite complicate, because of the coupling of the $\theta,\beta$ that coupling makes it untreatable.
![[Pasted image 20230316152649.png]]
```ad-tip
LEARN BY HEART
```


### Approximating parameter inference (LDA)
![[Pasted image 20230316152822.png]]
Since the problem is $\beta,\theta$ couplign we want a $Q$ that has no coupling of them, we've to disconnect the connection (make independent) the part of $\beta$ and $\theta$ ($\alpha$).

For sampling apporoach you can just infinitely draw sample, and will approximate precisely, more precision, more pain.

### Variational Inference

So we're choosing  a $Q$ that... we want to consider independant (all the RV), also each of them will have a parameter $\phi_k$
![[Pasted image 20230316153237.png]]

Of course this is not the true posterior, and we will be optimizing ELBO, we will plug this inside the expectation of $Entropy$ we've saw before.
![[Pasted image 20230316153347.png]]

### How the model change with this assumption?

This is our mean field approximation

(There is no $\beta$ in the first part is an error)

![[Pasted image 20230316153601.png]]


Gien a $\Psi$ the first derivative of the $\Gamma$ function we have.

![[Pasted image 20230316153857.png]]

## LDA Application

![[Pasted image 20230316154326.png]]

If you have collection of a lot of multinomial data? We want to represent visual contents. We can use blobs, or **Sift descriptors**! 
From lesson before, we get a lot of Sift Descriptor in a cluster of descriptors and then run $k-means$. This will return a codebook of descriptors and for each image, new data, we extract intrest point, do vector quantization and we construct a "bag of word" but with our intrest points (like words in the matrix) and then we with LDA can see 
![[Pasted image 20230316154949.png]]
Visual topic in an image, this is in a fully unsupervised way.

## Dynamic Topic Models

Sequence of models that evolves in time (unfolds), so we make them meet, hidden markov model. 
![[Pasted image 20230316155901.png]]

We can get thing like this
![[Pasted image 20230316160028.png]]

Or also: Well you know people win a noble in 60' for studying neuron, so... neuroscience is intresting
![[Pasted image 20230316160142.png]]

Also for structured.
![[Pasted image 20230316160234.png]]


### Take home lesson

```ad-summary


```


---
# References

**See paper : Variational LDA (Blei, Ng, Jordan) 2003**
