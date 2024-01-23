Date: [[2023-05-09]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Why is intresting Unsupervised learning with DL?

The problem is again modeling data distribution, by understanding direction of variation in the data, as we seen when we introduced [[Lesson 13 - Autoencoders#Manifold learning and assumption]]. Autoencoders will pop up again in a more refined way, because allows generating new observation understanding data and our model having understood the data can reason on them, we can see what can we do if we really understood nature of the data. For example predecessor relationships, or relationship between summation and multiplication.

Another problem is that having labelled data is costly and difficult, for example time series with forecasting is autolabeled, the future is to work on latent structure of data, discovering important features, learn task independently from representation, if we want to learn many tasks so multitask learning. 
Supervised learning representation gets easily corrupted when changed so the unsupervised are more robust under this.

## Why generative?

Focussing on density estimation approaches because if we have two modes in our data, simply applying interpolation would not capture those

### Approaching the problem from DL perspective

What we do is basically 

- Learn to model a density $P_{\theta}(x)$
- Implicit => Learn a proces that samples data from $P_{\theta}(x)$ $approx$ $P(x)$

## Taxonomy of Generative

So we can learn distribution from only visible RV or we can do inference on unobservable variables: [[Lesson 5 - Generative and Graphical Models]], usually distribution are tractable the visible one, for all the RV involved in the probability we have *observable* and *latent*, the latter are intractable because are continuous (no closed form for these integral). So we need to approximate, with *Variational*: Approximating the lower bound of the likelihood, and *Stochastic* approximation: learning by sampling.

Then we have sampling RNNs (CNN) these are [[Flow-based]], and normalizing flows is an interesting thing in generative, going from noise and backward and doing this with invertible transformations. Also these can be solved as differential equations.

For implicit models we have again stochastic with montecarlo approach, and GANs, where we don't have stochasticity system in deterministic.

## Learning with Fully visible information

If

$$
P(x)= \prod_i^N(P(x_i)| x_1,...,x_{i-1})
$$

We can compute from chain rule factorization, and if we have an image these $x_i$ are our pixels. So this is basically a Bayesian network.

In pixel RNNs we know that our pixels are ordered and we can process them by using a RNNs

So we give the first pixel to our network and then the images are generated pixel by pixel, we can also go in a frontier expansion and not only by scanning linearly.

RNNs are not good because the structure can not be parallelized and the time used if we want to generate pixel by pixel an image very big, would be very slow.

A better way to generate images is to use convolution so PixelCNN, we go from the layer below to the layer above, we lose dynamical memory from LSTM but providing enough convolution patches. The more suitable are [[Dilated Convolutions]].
These models are computationally not that effective, computational complexity is a cost, but the estimation of the density of the model is quite close to the effective distribution. These are effective density learner.

## Latent information

When we try to learn the $\theta$ parameterized model distribution

$$
P_\theta(x)=\prod_i^NP_\theta(x_i|x_1,...,x_{i-1})
$$

why obervable RV behave togheter? Rather than factorizing we introduce RV $z$. Since these is not tractable we will use neural network with latent variables.

## Neural Network with Latent Variables: Autoencoders

How do we train Autoencoders? These model have and hidden layer that learns representation and try to mimic the input.

Generating reconstruction from **sampled latent representation**, we could sample our $z$ from P(z) where this is a Gaussian distribution, then plug the $z$ into P(x|z), and generate, so training over parameters $\theta$, comparing P(x|z) with P(x). But we cannot backpropagate by sampling. We're maximizing the likelihood.

We're sampling our $z$ on a special Normal distribution based on $x$ it must be dependent on $x$

$$
z 
$$

We need a trick to solve non-differentiability. That's the reparametrization trick

## Reparametrization trick

Sampling isn't differentiable. We want to get a random sample by sampling with white noise and keep the sampling deterministic. So we put our $\mu(x)$ and $\sigma(x)$ like parameters and our $\epsilon\text{from}\mathcal{N}(0,1)$, so we backpropagate $epsilon$.

## Variational approximation

We will use the ELBO (Evidence Lower BOund), given an intractble lower likelihood we're maximazing that lower bound, as the difference between actual and approximated posterior. The posterior in this case is $P(z)$, so we're approximating that with the decoder.

We've expectation ran over $Q$, where the `Q` is a neural net.

The term before the `+` is a `reconstruction error`: that is solved because we can use the reparametrization trick, that expectation isn't ran over $Q$ but over the normal, the net provied the parameters, instead the term after is the KL-divergence, because we're approximating that distribution, and if $Q$ is a net and we push it to approximate that normal, the fact that we're keeping the thing simple is doing regularization. Where the distribution is a normal with high probability at the center and then smoother in a radious.

## Variational Autoencoders

Decoder network : $P(\tilde{x}| z, \theta)$, while encoder network $Q(z|x, \phi)$ with $\phi$ parameters.
[insert image]

Variational parameters encoder and likelihood parameters in the decoder.

### VAE training

We're just training by backprop on $\theta$ and $\phi$ to optimize.

[image]

We're pushing the learned normal to be that similar to $P(z)$ the actual distribution, but we decide how the latent space is shaped. But we can make it different by changing with mixture of gaussian or a poisson. This happens to be the simple and the most regularizing one. Maybe we will lose the fact that without simple gaussian is closed form but we can approximate.


### VAE final loss

 [image]

This loss can be seen under an information theory perspectic

[insert image]


### VAE Vs Denoising and Contractive

In VAE we're drawing from a distribution that expands with different probabilities and we're exploring by sampling and a smoothing coverage (without holes) will give us a regularized model, also the quality of the latent space will be better (without holes), also in contractive we have point where we know nothing instead in  

## Conditional Generation (CVAE)

What if we want to learn $P(x|y)$?

[image]


What we would like to do is really [[Disentangled Learning]].

























### Take home lesson

```ad-summary


```


---
# References

