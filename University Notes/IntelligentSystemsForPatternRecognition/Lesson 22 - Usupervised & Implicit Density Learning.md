Date: [[2023-05-10]]

Status: #notes

Tags: #ispr

# Sampling is all you need

In the variational autoencoder of yesterday there were parameters for likelihood and those for distribution.

If we want to generate high quality images comparing pixel by pixel isn't a smart idea (euclidean distance), on average less wrong, but sampling from VAEs isn't good.

Comparing images, or worse, videos, how do we define good reconstruction?

Well just focus on generating samples, in the taxonomy we refer now to *Direct* approach were we are fully determinstic and the stochasticity is inside the network.

![[Pasted image 20230711193153.png]]

## Learn to generate samples

Remember when we talk about two way to approaches in [[Lesson 21 - Unsupervised & Generative DL I - Explicit Density Learning#Approaching the problem from DL perspective]] we're in the second part implicit density learning!

![[Pasted image 20230510162655.png]]

If we want to learn to do this we are trying to sample from very high dimensional spaces, for example images with a lot of pixels.

## GAN catch

We're going to the source, directly learning the sample procedure more than learning the distribution!

We go from a Gaussian sampling to actual real sample

![[Pasted image 20230510162834.png]]

Starting from random noise and then we have a network that transform random noise in *bad samples*.

Now we want to do improve reconstruction quality with a two step architecture. Now we align another model that discriminate *Discriminator*, this is a simple binary classifier that can judge if the data is from data or generated

![[Pasted image 20230510163036.png]]

So is the input real or it's fake? We would like to knwo from what branch comes the input.2

We woul like with *generator* to create better reconstruction that tricks the *discriminator*, this is how we want behave these network.

We formalize this behaviour as a [[Nash Equilibrium]] problem. One is maximizing a function the other minimizing, the same function, this give rise to a [[Zero-sum game]].

Our loss will back propagate from Original/Fake -> Discriminator -> Generator G(z) -> Random Noise all is differentiable

What is this loss saying? 

![[Pasted image 20230510163316.png]]

This is a composed loss let's decompose it:

- The first part estimate ability of discriminator (variation of the crossentropy), so the job of discriminator is to maximize $\theta_D$,  so become very good to make that first term really high, where $E_x$ is the expectation over data (Log of discriminator output where the output is usually or 0 or 1 for fake and real. So discriminator want to push high the first term and low the second (it has a minus) in order to maximize the whole expression.)

- While the $E_z$ is the data drawn over the normal, we pick those data plug into the $z$, and discriminator want to put that to zero (second part) if wants to maximize, so discriminator update parmeters to fire $1$ if input is from data and to say $0$ if comes from random distribution, at the same time we're minimizing over parameters of the generator $\theta_G$, because we want with generator to make discriminator fire $1$ instead of 0.

Getting in optima is really difficult, sometime we get stuck in circular point around the saddle point.

We need to alternate between step 1 and step 2, ascending with discriminator and descending with generator.

![[Pasted image 20230510163745.png]]

One of the key issue is that optimizing the generator doesn't work well usually, because the cost function is *flat*. Because depends of the discriminator, if it's is very convinced at the beginning of the training discriminator become very good fast to learn, to we're not getting learning from generator after those steps. At the beginning was very difficult train these models, were unstable, there was heuristic, and the fact that discriminator is skilled is difficult to control. 

## Issue and solution

![[Pasted image 20230510164158.png]]

Instead to use minimax cost we could use a slightly better formulation, which has always some gradient, so we're changing the loss tha is difficult for the discriminator

That thing $C_G$ is maximizing the ability to fool our discriminator, so we basically change only the loss of the generator to be the green one.

## Training GANs

![[Pasted image 20230510164342.png]]

There are two steps for discriminator and generator.

We sample minibatches of data and noise, (expectation: expected gradient)

- The first is the max part of the loss
- The second is the min part of the loss

## Yet another problem: Mode collapse

![[Pasted image 20230510164527.png]]

Another problem here in training initially were the **mode collapse**. The generator instead of generate samples that have different modes so from empirical distribution, instead started to generate almost similar samples realistically looking but a little diverse. So generator becomes good at generate subsets (different modes).

Theoretically the distribution learned by generator and empirical distributon. If we take KL divergence (not properly that but a measure of similarity between distribution) the two distributions can be tought of components.

We have two components:

1. A quality component: Two distribution are similar if sample are similar in quality: This depends on generator. 
2. A covering, overlap of the two distribution, so how much the generated one cover the empirical. This doens't depend on generator

And the method of optimizing has no gain to have better coverage only incentive the quality part. And you get mode collapse!
The second part of the divergence measure the overlap between the two distribution, so how the two cover the original. Generator doesn't have incentives to have coverages, so get better quality to generate certain high quality samples with not a lot of diversity.

## Wasserstein Distance Models

So we pickup different loss the **Earth mover distance**, where we can estimate how much work we need like *edit distance* but for distributions. So we are minimizing what is inside the bracket with expectations. Samples over real data and samples over the made up ones, so aligning real and fake data distance (minimizing it). That's a lipshitz constraint linearly. Gradient are bounded by a constant *Lipishitz constant*, so our model is contractive. 

![[Pasted image 20230510164906.png]]

This is a constrained optimization problem but there is an easy way to make it work, optimizing the discriminator and generator by clipping the discriminator would be lipshitz to stay in that region.

Now we're decreasing the learning but making it stable.
![[Pasted image 20230510165419.png]]

Looking at it as optimization problem 

![[Pasted image 20230510165452.png]]

The generator has no gradient but WGAN loss has gradeint everywhere and work nicely.

For images we have this architecture

![[Pasted image 20230510165617.png]]

A GAN is a deconvoltion-convolution newtork, basically swapping the convolution-deconvolution we've seen before, but deconvolution is just convolution as we saw, we upsample. Deconvolutional model generating and convolution to discriminate.

## Latent Space Arithmetic

In VAE there is not semantic space, here we can alter some sample that are not far they have a reasonable relatioships between them, these are three samples if we interpolate we get this. There is StyleGAN that extract from latent code in style and content

![[Pasted image 20230510170033.png]]

## [This people do not exists](thispeopledonotexists.com)

![[Pasted image 20230510170113.png]]

## Progressive GAN

We start to learn very dumb images and then train toward more difficult images, for example 4x4 images, then we try to generate on top of that an 8x8, every time we train everything without freezing layers.

![[Pasted image 20230510170309.png]]

## Smooth transitions

Since we're training everything we don't want to destroy all we pre trained. We train 32x32 but the image learned by 32x32 is generated with some help from the previous 16x16, using that $\alpha$ at the start so the most effor is done from that part initially this is called **Smooth Transition**. After we're done with this transition (residual connectivities) we discard them. 
The idea is to use the things we learned before to help learn the higher resolution part and then add some decay to $\alpha$, so that it learn on it's own the right thing.

Smooth transition is good also for

![[Pasted image 20230510170414.png]]

## Conditional Generation

What if want to generate sample that have males, femals and age?

![[Pasted image 20230510171635.png]]

We can change the conditional variable, for example the age. We input to the generator the age range, for each image we need the age of the person, and then we feed those informations. There are some issue like aligned data. And we try again to disentangle $z$. The $y$ capture stylistic and $z$ accounts for the randomness, for example $z$ changes identity, while $y$ changes the hair, eye color and so on.

Conditinals can be images

![[Pasted image 20230510171937.png]]

## Style transfer

Monet to Van Gogh: Building without a lot of aligned data. We don't have a lot of images for that task.

So we have a cycling between generator that goes from an image like conditional infomration, generator generates the image horse to zebra for example. Then we use another generator  that enter as condioning image the one transformed. We impose some losses. One in the domain $A$ discriminate if real or not, then second part of the loss, check if the two images (two horses are similar, so the one we picked up and the one we generated back). If you're good to do style translation we have to be good enough to reverse it. This is asymettric, there is not cycle. The other bit get the generated image horse, generate a zebra and uses the original zebra and discriminate. This closes the loop. No need for aligning data. There are a lot of adversarial losses. 

There are three components:

One is: Discriminators trying to guess if images in domain A or B are real
The other two are: Check that images that check that images of domain A translated to B and back to A have to be identical.

CycleGAN is at the base for StyleGANs in literature. StyleGAN add components to control styles.

![[Pasted image 20230510172107.png]]


Famous sample is Putin riding zebra, and we have putin becoming a zebra

![[Pasted image 20230510172810.png]]


## Adversarial Autoencoder

Best of two words!

Why is this good? Well we don't have to compute the KL divergence, that for Gaussian is straightforward but if we want to put (as prior) a very strange distribution it becomes impossible to have in closed form, this allows us to impose some distribution that are non gaussian in sample space! I can have 

#### Mixture of gaussian

Using a discriminator that have to decide from GAN generator and the $z$ drawn, this is quite helpful and reconstruction. Also Mixture of Gaussian (flower shaped chart) has closed form solution.

![[Pasted image 20230510172838.png]]

### AAE Vs VAE

The mixture of Gaussian B,D.

For example song autoencoder : input is music, the $z$ value is compressed version of the song, we have auxiliar information about it, and we want to leverage the fact that in music there is a structure in classical and punk music there are differences. So one petal is dedicated to a genre and another petal far is classical. But at the center of the flower we have that there are songs that reinterpret classical music in punk term! This cannot be imposed by KL divergence. Is more flexible

![[Pasted image 20230510173052.png]]

### Training AAE

Well get rid of KL and use adversarial loss like Wasserstrain!

![[Pasted image 20230712124713.png]]

## AAE - Style transfer

![[Pasted image 20230510173450.png]]
The identity information is captured what remains is the style that is forced to being captured, if we want to change style we sample and keep $y$ fixed.

### Take home lesson

Learning to sample if a good move if we want realistic samples we don't care about the distribution, no information about that distribution cannot explore it. No nice structure latent space. For representational learning we use AE or AAE. GANs are unstable and difficult to train, until we use Wasserstrein loss.

```ad-summary


```


---
# References

