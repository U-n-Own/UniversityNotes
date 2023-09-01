Date: [[2023-05-09]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Why is intresting Unsupervised learning with DL?

The problem is again modeling data distribution, by understanding direction of variation in the data, as we seen when we introduced [[Lesson 13 - Autoencoders#Manifold learning and assumption]]. Autoencoders will pop up again in a more refined way, because allows generating new observation understanding data and our model having understood the data can reason on them, we can see what can we do if we really understood nature of the data. For example predecessor relationships, or relationship between summation and multiplication.

Another problem is that having labelled data is costly and difficult, for example time series with forecasting is autolabeled, the future is to work on latent structure of data, discovering important features, learn task independently from representation, if we want to learn many tasks so multitask learning. 
Supervised learning representation gets easily corrupted when changed so the unsupervised are more robust under this.

## Why generative?

Focussing on density estimation approaches because if we have two modes in our data, simply applying interpolation would not capture those. A generative approach is more incline to learn and really understand the data, because it need to reconstruct them! So understand how data vary and when data is normal (detect anomalies).

### Approaching the problem from DL perspective

What we can do are basically two approaches 

1. Learn to model a density $P_{\theta}(x)$
2. Implicit => Learn a process that samples data from $P_{\theta}(x)$ $\approx$ $P(x)$

## Taxonomy of Generative

![[Pasted image 20230711115413.png]]

So we can learn distribution from only visible RV or we can do inference on unobservable variables: [[Lesson 5 - Generative and Graphical Models]], usually distribution are tractable the visible one, for all the RV involved in the probability we have *observable* and *latent*, the latter are intractable because are continuous (no closed form for these integral). So we need to approximate, with *Variational*: Approximating the lower bound of the likelihood, and *Stochastic* approximation: learning by sampling.

Then we have sampling RNNs (CNN) these are [[Flow-based]], and normalizing flows is an interesting thing in generative, going from noise and backward and doing this with invertible transformations. Also these can be solved as differential equations.

For implicit models we have again stochastic with montecarlo approach, and GANs, where we don't have stochasticity system in deterministic.

## Learning with Fully visible information

If

$$
P(x)= \prod_i^N(P(x_i)| x_1,...,x_{i-1})
$$

We can compute from chain rule factorization, and if we have an image these $x_i$ are our pixels. So this is basically a **Bayesian network**.

In pixel RNNs we know that our pixels are ordered and we can process them by using a RNNs

So we give the first pixel to our network and then the images are generated pixel by pixel, we can also go in a frontier expansion and not only by scanning linearly.

RNNs are not good because the structure can not be parallelized and the time used if we want to generate pixel by pixel an image very big, would be very slow.

A better way to generate images is to use convolution so PixelCNN, we go from the layer below to the layer above, we lose dynamical memory from LSTM but providing enough convolution patches. The more suitable are [[Dilated Convolutions]].
These models are computationally not that effective, computational complexity is a cost, but the estimation of the density of the model is quite close to the effective distribution. These are effective density learner.

![[Pasted image 20230711115744.png]]

## Latent information

When we try to learn the $\theta$ parameterized model distribution

$$
P_\theta(x)=\prod_i^NP_\theta(x_i|x_1,...,x_{i-1})
$$

"Why obervable RV behave togheter in such a way?"

Well in order to model this joint distribution we have to introduce some "explainatory" variables that are reponsable for the visible ones, so we introduce RV $z$ in order to explain the *relations between visible RV*: that's the job of **latent variables**.

A latent process introduced by marginalization.

$$
P_\theta(x) = \int P_\theta(x | z) P_\theta(z)dz
$$
Approximation: Variational or Monte Carlo.

And since these is not tractable we will use neural network with latent variables: Autoencoders!

## Neural Network with Latent Variables: Autoencoders

How do we train Autoencoders? These model have and hidden layer that learns representation and try to mimic the input.

So we would like to approximate the $P_d$ distribution (decoder). We are going to learn two distribution the encoder taking $x$ and mapping to $z$, so learn the posterior (probability of latent given visibile that's by definition) and decoder map the istance of $z$ to reconstruct a new $\tilde x$. So far we did it intuitively.

![[Pasted image 20230711122310.png]]

Well we can think to that integral as an *Expectation*. We're speaking about decoder part now.

![[Pasted image 20230711142401.png]]

Generating reconstruction from **sampled latent representation**, we could sample our $z$ from $P(z)$ where this is a Gaussian distribution (because Gaussian is simple) to generate our latent variables, then plug the $z$ into $P_\theta(x|z)$ (parameterized!), and generate, so training over parameters $\theta$, comparing $P_\theta(x|z)$ with $P(x)$. 

All beautiful, but we cannot backpropagate when sampling. We're maximizing the likelihood.

## The catch behind VAE

![[Pasted image 20230711143103.png]]

We're sampling our $z$ on a special Normal distribution based on $x$ it must be dependent on $x$

We need a trick to solve non-differentiability. That's the reparametrization trick

So for the intractable distribution we will use Variational Approximation and then we will solve the backpropagating issue of sampling with reparametrization.

![[Pasted image 20230711143424.png]]

## Reparametrization trick

What is reparametrization trick?

Reparametrization trick is used in VAEs because when we want to learn the distribution P(x) we will approximate by sampling from P(z) gaussian distribution, in order to minimize KL divergence between P(x) target and P(x | z), since we're doing backpropagation we need a way to backpropagate trough sampling, we can't take the derivative over a stochastic process, so what we do is in order to get our z from Gaussian(mu,sigma), we parameterize mu and sigma and we sample an epsilon from a white noise normal (0,1) and then multiply by sigma this epsilon and add the mean, this is equal to z we wanted."

![[Pasted image 20230711150027.png]]

Sampling isn't differentiable. We want to get a random sample by sampling with white noise and keep the sampling deterministic. So we put our $\mu(x)$ and $\sigma(x)$ generated deterministically, like parameters, and our $\epsilon \sim \mathcal{N}(0,1)$, then we do this

$$
z = g_{\mu,\sigma}(\epsilon) = \mu + \epsilon \sigma
$$

![[Pasted image 20230711164851.png]]

### True form of reparametrization 

Remember that we're taking our input $x$

![[Pasted image 20230711165308.png]]4

so we take out the randomness from our backpropagation.

## Variational approximation

We will use the ELBO (Evidence Lower BOund), given an intractble lower likelihood we're maximazing that lower bound, as the difference between actual and approximated posterior. The posterior in this case is $P(z)$, so we're approximating that with the decoder.

We've expectation ran over $Q$, where the `Q` is a neural net decoder, this will be train to optimize a part of the loss function (reconstruction part). The encoder will play role of $Q$, will optimize the gap between $Q$ (what is outputting) and $P(z)$.

![[Pasted image 20230711171847.png]]

The term before the `+` is a `reconstruction error` telling how good **decoder** is to generating $x$ that are likely: that is solved because we can use the reparametrization trick, that expectation isn't ran over $Q$ but over the normal ($z$), the net provied the parameters.
Instead the term after is the KL-divergence between $Q$ and $P(z)$, because we're approximating that distribution, and if $Q$ is a net and we push it to approximate that isotropic normal $\mathcal{N}(0,1)$ that is $P(z)$, the fact that we're keeping the thing simple is doing regularization. Where the distribution is a normal with high probability at the center and then smoother in a radious. And restraining to a simple normal distribution we ensure that our latent space captured will be the simplest possible, so we won't have strange distribution that leads to overfit in the latent space. Using that is a *regularization* to latent space!

## Variational Autoencoders

Decoder network : $P(\tilde{x}| z, \theta)$, while encoder network $Q(z|x, \phi)$ with $\phi$ parameters.

![[Pasted image 20230711175202.png]]

Variational parameters in encoder matching $P(z)$ (gaussian) and likelihood parameters in the decoder (generating x that are likely). We propagate $\theta$ and $\phi$ two neural nets.

### VAE training

We're just training by backprop on $\theta$ and $\phi$ to optimize.

![[Pasted image 20230711185004.png]]

We're pushing the learned normal to be that similar to $P(z)$ the actual distribution, but we decide how the latent space is shaped. But we can make it different by changing with mixture of gaussian or a poisson. This happens to be the simple and the most regularizing one. Maybe we will lose the fact that without simple gaussian is closed form but we can approximate.

### VAE final loss

You make the encoder being network that predicts one value for the mean and one for variance, use reparametrization trick to generate actual $z$ and backpropagate with this error.

![[Pasted image 20230711185340.png]]

This loss can be seen under an information theory perspectic

![[Pasted image 20230711185628.png]]

## Usage of VAE?

Well as soon ELBO flattens throw away the encoder and use the decoder to generate samples

![[Pasted image 20230711185838.png]]

### VAE Vs Denoising and Contractive

In usual denoising autoencoders we we're doing similar thing in a non probabilistic manner. We were giving a broken input and training them to retreive them to a fixed state.

In VAE we're drawing from a distribution that expands with different probabilities and we're exploring by sampling and a smoothing coverage (without holes) will give us a regularized model, also the quality of the latent space will be better (without holes), also in contractive we have point where we know nothing instead in  

![[Pasted image 20230711192204.png]]


## Conditional Generation (CVAE)

What if we want to learn $P(x|y)$? We give an image and a bit encoding some information about image, like sex of a person. We're biasing to focus on something different.

![[Pasted image 20230711192532.png]]
What we would like to do is really [[Disentangled Learning]].

























### Take home lesson

```ad-summary


```


---
# References

