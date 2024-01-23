Date: @today

Status: #notes

Tags: #ispr

## Small introduction about Diffusion

```ad-important
The essential idea behind Diffusion is to systematically and slowly destroy structure in a data distribution trought an interative process: Forward Diffusion. We then learn how to reverse this process that restore that structure in data, yielding a highly flexible and tractable generative model of our data.
```

But from where *Diffusion* comes from? Inspired but non equlibrium thermodynamic from physics, so system that are not in termodynamic equilibrium: eg drop of paint in a bucket of water.

![[Pasted image 20231026171723.png]]

At the start we have some very complicated information then this will tend to spread, so basically reverse Entropy

# Why diffusion model?

Be progressive when generating images for high resolution

![[Pasted image 20230511142834.png]]

These are latent variational in the taxonomy we've seen, now the size is the same in input and output, similar to what happens in flow base models. When we introduce latent stuff we have an intractable part, we're going to use ELBO.


## Two step models

We again can interpret ad encoder-decoder architecture si incremental addition of noise, but we need some properties to this noise addition. We're intested in the reverse part, the second part is when training happens the forward noising supposed "encoder" ins't trained.
![[Pasted image 20230511143057.png]]

### Forward diffusion - Intuition

![[Pasted image 20230511143311.png]]

These corrupted $z$ can be seen as a latent representation of our data. Using gaussian noise until we get to $T$.

### Noise addition

![[Pasted image 20230511143418.png]]

Where the noise is that $\epsilon$. $\beta$ hyperparameter is an index targeted for each time step, his job is to regulate how much noise we're adding at each time step. High frequency information is the first that get's corrupted so high quality images is destroyed.

Since we're corrupting by time step and choosing the $\beta$ to decide how much corrupt, this is much like a markov chain where we go from a pixel space to another

### Distributions

So $x$ to $z_1$ and so on are conditional markov chains, and the $q(z_1,...,z_T|x)$ is the conditional, giving us the clean part for the forward process. This give rise to a problem, inefficiency.
![[Pasted image 20230511143750.png]]

This is sequential, to reach $z_t$ we have to get trough all the other noisy samples.

## Diffusion Kernel

$\beta_t$ are fixed now, we're changing the $\alpha$. We draw a sample from there to get diffused gaussians.

Now we've got a way to writing our marginal, introducing by marginalization $x$.

![[Pasted image 20230511144216.png]]

## Evolution of diffused data distributions

Now we can visualize our distribution initially is very complex, every time we apply noise we simplify this distribution, coming to the most common one, a gaussian $\mu=0$ with $\sigma=1$.
![[Pasted image 20230511144458.png]]

## Denoising

A gaussian is easy to sample, so we sample from that, then we apply the reverse distribution in an incremental way. To decode back to our data space.

Now we can't evaluate that distribution because of denominator and first part of numerator, because the the integral we've seen up.

![[Pasted image 20230511144636.png]]

## Reversing process

So what we're doing now is to approximate considering $P_\theta(z_{t-1}|z_t)$ to be a normal, $\sigma$ there is a scheduler making more o less noise. But what the network needs to learn is how to output the mean of a gaussian.
![[Pasted image 20230511145035.png]]

## Training

![[Pasted image 20230511145635.png]]

The first sample $P_\theta(z_{t-1}|z_t)$ is easy, is neural net predicting the mean, for the rest we use ELBO

![[Pasted image 20230511145823.png]]

The first term as always is reconstruction: how much good is the model to approximate our data from the distribution we want to approximate. The second term is Kullback Liebler divergence that want to make similar the two distributions. One we know is a normal distribution the other also can be said that is normal. KL can be solved in closed form if the two are normal and so we can think to this that is a distribution that knows what noise is being added.

## ELBO Loss Function

The true denoising distribution can really reconstruct the noise and push the predicted one with the mean $\mu_\theta$ to match the actual mean. On training data $x$ is known. We're forcing the model to match that statistichs.

![[Pasted image 20230511150138.png]]
There is a simpler form of equation for the loss, people realized that if we reparametrize how we write things.

The data can be recovered if we know the noise

## Training: Practical view

Instead of predicting noise at time $t$ we predict *what nosie* is being applied at that time, so we change how we learn stuff, isntead of predict average noisy image at time $t$ we just learn the noise we need to take out.

This becomes very easy because it's a MSE minimization. This ELBO is quite close to the original log likelihood and very easy to interpret.

![[Pasted image 20230511150521.png]]

## Implementing training

![[Pasted image 20230511151936.png]]

Sampling from the minibatch, the cost of training is very simple, the true cost is when we decode!

### Diffusion images model

![[Pasted image 20230511152248.png]]

For the time we use some encoding much like [[Lesson 11 - Attention and Transformers#Position repreentation with sinusoids]].
We can insert time as another channel into the network or we can combine at every channel by shifting the result of convolution and then shifting these activations. There are a lot of parameters to upconvolve in time...
Also for conditional generation this is useful.

### Noise schedules & tricks

Now what is used for diffusion are U-net with self-attention, a good thing is to have many steps or be smart to jump steps because we have to be careful when noising and denoising. Idea is to *slowly increase the noise*, careful at the beginning because we don't want to destroy the high frequency information, but at the end we can hammer with high beta.

We can as well learn $\sigma_t$ by plugging it in as an another parameter, and also $\beta_t$.

![[Pasted image 20230511152546.png]]

## Guidance Classifier

![[Pasted image 20230511153137.png]]

This is something that behaves like a momentum, not going in random direction with nosing process, but we know that image generating is something like a flower for example, so we insert into parameters the gradient direction of that class that is predicted, so that model know how the standard flower should look like, then we have the way to change the style of the flower.

![[Pasted image 20230511153418.png]]

This is telling us : give me an image knowing a priori that the image is from a certain category, how do we get that probabilty? We mix two distribution, the $\gamma$ scalar is a tradeoff from the creative part. Big $\gamma$ gives us a very good looking flower but similar to each other, while decreasing it make it more different but decrease the quality. Think to this like a *temperature*. 

Problem with gradients is that they are unstable because if we are in a certain part of the denoising we cannot know if that's a flower or not. Noise can't be classified...

### Classifier free guidance

Derivative of conditioned model is given like a mix between **unconditional distribution** (uncoditional diffusion model) and the **conditional diffusion score**, $c$ is in input to the thing that return the noisy image. This tells us that we can mix diffusion model without auxiliar information and one trained with auxiliar. 

We can train the same model to behave conditionally and uncoditionally by playing with dropout. And then we play with our choosen $\gamma$, if $\gamma$ is zero the part disappear and vice versa for zero, for higher than 1 is more conditional and trust less the uncoditional... 
![[Pasted image 20230511153826.png]]

### Guidance high resolution

Diffusion models after the other independently trained that upsamples using guidance introduced like "time", there is a channel for the U-net that encode this.
![[Pasted image 20230511154212.png]]

### Denoise with semantic segmentation

This is usually a supervised model, but now is all unsupervised!

![[Pasted image 20230511154345.png]]

### How do we generate images out of text?

What is the conditional information? Well the text is something that can be seen in a vectorized way.

![[Pasted image 20230511154518.png]]

Each channel of the U-net receives a part of the text, using embeddings from [[Lesson 11 - Attention and Transformers]] LLM.
![[Pasted image 20230511154529.png]]

### Latent Space Model

As we said this is working in a latent space same dimension input and output, but if we deal with video this becomes a nightmare, so what we do is to project into a much small latent space like when we spoke about [[Lesson 13 - Autoencoders]], and so on we get this.

We apply the denoisign reverse process, denoising the vector, having and encoder and decoder on top of this.
![[Pasted image 20230511154647.png]]

## How does DALL-E works?

They use CLIP : encode visual and text information where the two modalities are semantically aligned, this favours the fact that the scalar product of embedding of text and image are aligned, and then pick images and text that have nothign to do with each other and train with a contrastive way-like. The part of the model that learn the encoder from text to clip space.

So encode text into a CLIP, then with an autoregressive (pixel to pixel or RNN) or diffusion process we generate the image CLIP, and then we apply the diffusion from the latent space (CLIP) generate the image.
![[Pasted image 20230511155114.png]]

### Take home lesson

Important thing from VAEs is that we have same size of input and output, then we're going in timesteps, if we define in continuous time we have a VAE with infinite layers, we can solve a continuous time differential equations pushing these model to the limit.

GANs and VAEs can be use togheter to solve the problem to cover the entier distribution to help the discriminator by adding noise to discriminator and generator, diffusion model have the problem when converting backwards. What if we can train GANs to jump 10 - 20 step the diffusion process, so by doing faster denoising.

```ad-summary


```


---
# References

