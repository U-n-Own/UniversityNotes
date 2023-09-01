Date: [[2023-03-29]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Aka The first and last deep learning model

By chance this was one of the first deep learning model ever done, autoencoders was used as denoisers for example, these architectures were proposed as early neural networks structures for dimensionality reduction.

*Deep neural autoencoder* are used for representation learning
- Sparse approach
- Deonoising approach
- Contractive approach

*Deep generative-based autoencoders*
- Deep belief nets
- Deep BM, like we've seen use of Boltzmann machines earlier.

## Basics of autoencoders

Project our input in an hidden space that can be smaller, this was the initial intuition, latent space projection. 

Latent variable = Latent neurons

The first part is the encoder that *encode* in low spaces information and the *decoder* try to output them in a larger dimension again.
![[Pasted image 20230329162400.png]]

These are basically compressors of information, we've an information bottleneck for the reduction of hidden layers dimension, by making it sparsely active (sparse autoencoder), then our training is very dumb, minimizing discrepancy between the two function that goes from big to small and small to big. We basically pick the important information with lossy compression.

## Neural autoencoders

Well linear autoencoders are basically another way to do PCA, these neural has some nonlinearity that introduce possibility to learn non trivial identities and can be regularized

## Sparse autoencoder

Add a term to the cost function to penalize the hidden layer, so that units that are active are smaller.

The function $\Omega$ depends on what we want to learn, for example in this case is the $L_1$ Norm, sum of the absolute values. Usually regularization used directly on matrix of weight not on the **Activation**

![[Pasted image 20230329163058.png]]

### Probabilistically speaking

First term there is our likelihood, and the second is a prior, but that prior isn't a real prior, we plug data into $h$.

So we would like to control how this behaves, and we can see that $P(h)$ is basically like the $L_1$ norm.

Assuming that hidden activation are distributed like a Laplacian distribution.
![[Pasted image 20230329163250.png]]

## Denoising autoencoder (DAE)

Our $\hat{x}$ is a noisy version of $x$, so just applying some gaussian noise to our $x$, so basically we take the data, add noise.

If we loop this way what happen? We start to sample from dataset, sample a noise add them and feed to the autoencoder and ask the autoencoder to return the unnoised one, everytime we do this we add different noise, then we will pickup another sample already seen at second epoch, but this time we have a different noise on the data, so we can say that we're stochastically working on the data, we're not learning $x$, but we're working to learn some representation of our data.

![[Pasted image 20230329163632.png]]


### Another intrepretation by probability

What the autoencoder learnt is to denoise our samples, if we add deterministically noise we can learn to denoise. This is basically learning a distribution, a denoising distribution. These are like probabilistic autoencoders! We can think to this like how diffusion models work.

![[Pasted image 20230329164038.png]]

## DAEs as Manifold Learning

When we try to learn representation we're doing some assumption the main one is the

```ad-important
Manifold Assumption: Our data, how much complex, high dimensional they are, won't occupy even a little bit of the total space!
```

Most of the time assigning values to a pixel randomly gives us only white noise, the space of images is a very small subset, these are called ***Manifolds*** lower dimensional set embedded in high dimensional space, so moving in a certain direction gets us white noise but moving in good direction gives us images that are intresting.

Locally these looks euclidean, instead globally these spaces are not euclidean (manifolds). That's what we want autoencoder to learn! To move on variation of data that are meaningful, so only the direction that make us stay in the manifold generates compliant data, for example take the MNIST, if we go along that line we find 1s and if we get away we get more noisy pictures, for example, if we have our data $x_i$ where we can find $\hat{x_i}$? Well at least out of our manifold, so what we're learning is to bring our *noisyied* version of the data to get as fast as possible toward the original $x$, how do we do it? By going on the green line.

So what we do with autoencoders is to a lot of different version of noisy data, and bringing it back by following green arrow, the green arrow are like *force fields*, bringing the particles to stable points, so we learn to approximate the score (Score matching aka Gradient of the log of the likelihood), the degree of proprotionality will be how much we sampled.

$g(h) - x$ is the difference between the noise and our original data, take an input and a random vector, the autoencoder will recreate an exact digit on the manifold by iterating process. This is basically sampling process, this is at the basis of diffusion models, GANs and other thing. So by start from white noise we train a neural network to do this, learning the vector field. So if we sample some random noise we will get a proper data, but with [[Lesson 23 - Generative III - Diffusion Models]] we can target what data we want to reach. 

![[Pasted image 20230329164243.png]]

#### Manifold learning and assumption

![[Pasted image 20230329170244.png]]

Assuming our data lives on these manifolds we're moving on a single direction can change how the number is rotated maybe, or maybe another direction on our manifold is the shrinking. 

[[Disentangled Representations]] are very hard to learn, if we change a single direction maybe from a 4 we're shrinking, rotation and becomes similar to a 6 for example. So disentangling the direction is surely useful because maybe going in a single direction make changes in only one thing in our image.

We're desperately taken these noisy thing and find a way to encode them in a neural activation hoping that neurons capture the manifold structure, manifold does this explicitly, other models build implicitly on this assumption

## Contractive autoencoder

Another approach similar to sparse autoencoder.

The whole autoencoder can penalize in output, but infinitesimal variation in $x$ have not to produce big variation in $f(x)$.
This is connected to denoised autoencoder and what is actually doing.

![[Pasted image 20230329172657.png]]

## Stacking deep autoencoders

First big deep network were autoencoders, for example we have our $h_4$ disentangled representation of our data, and instead doing directly from $x$ to $h_4$ we have middle layer, in the past going trough many layers was difficult, now is not!

![[Pasted image 20230329172943.png]]

## Unsupervised layerwise pretraining

For autoencoders we want to build them deep, at the time backpropagate trough many layers we do a clever trick, we learn the first layer then we use the weights of the learnt architecture we fix them so we don't need to backpropagate and we go for the next.

![[Pasted image 20230329173914.png]]

![[Pasted image 20230329173928.png]]

Rearrangin this becomes: A RBM!
![[Pasted image 20230329173944.png]]

## Deep belief networks

![[Pasted image 20230329174007.png]]

## Deep boltzmann machines

We're taking arrow going and coming from the layers, so we want the $P(h_1)$ depending from the upper layer $h_2$ and the lower layer $x$.

![[Pasted image 20230329174035.png]]

By marginalization on $h_2$, we get two distribution one coming from
![[Pasted image 20230329174212.png]]
![[Pasted image 20230329174900.png]]

## Application of AEs

Take vector and project it with autoencoders, our autoencoder will cluster them.
![[Pasted image 20230329174923.png]]

## Autoencoding sounds
![[Pasted image 20230329175012.png]]

## Black and white as noise

Going from graylevel to colored pictures!
![[Pasted image 20230329175156.png]]

## Multimodality

Imagine to have a sound, or an image, or a text. We can us the pre training trick that has a compact representation at low level and fuse them at high level, when we have processed an enough semantic representation, representing this. By calling 1 : Image and K : text, we can do: Input an image sampling away with boltzmann machine and decoding from the image to text! So obtaining a set of words with highly likely associating these models with images! Or pluggin in some tags, word and getting from them an image representation. Similarity search that search on manifolds.
![[Pasted image 20230329175234.png]]

## Anomaly detections
We get the parameters, if the error we reconsturcted from the anomalous piece of data we can detect anomalies.
![[Pasted image 20230329175751.png]]

### Take home lesson



```ad-summary


```


---
# References

