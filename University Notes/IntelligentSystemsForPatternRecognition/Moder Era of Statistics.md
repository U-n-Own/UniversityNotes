Date: [[2023-05-06]]

Status: #notes

Tags: [[MLOuterSources]]

Notes taken from presentation of lesson 9 of MIT 6.S191 course link here 

In the last years we noticed that by increasing the power consuption we get better and better results, true for generative models that in 2014 were really bad like GANs and now we got Stable Diffusion like models that outperform humans in art. Scaling is a key, but should keep scaling because bigger seems better but why?

Some numerical analysis on MNIST reveals that we have 60k points on 28x28 images, models today have millions of parameters and are trained on MNIST, performance improves with increasing parameters, but what are we learning, from these 60k images?

Generalization bound $\alpha$

$$
\alpha = \sqrt{\frac{\text{\# of params}}{\text{dataset size}}}
$$

Why this happens?

## Double Descent

Well we need to take a look at a phenomena names [[Double Descent]] more on this can be explored in this 
blogpost [Double Descent in Human Learning](https://chris-said.io/2023/04/21/double-descent-in-human-learning/).

Look at the puple thing at the end, look that at a certain point performance is optimal and if you increase the size you overfit on the data, but then we get a second descent. This has been observed many times in different models

![[Pasted image 20230506172358.png]]

We'll call that region [[Overparametrization Regime]], in there a different behaviour emerges
![[Pasted image 20230506172933.png]]

What is robustness in DL? Control error in the output if we have a little change in the input. We have theory for this

But not everything improves, problem of bias and accuracy on minority sample worsen in accuracy.

Reasoning (ability to plan and so on) stays unchanged wiht respect to the parameters, unless you give the model a simulation engine, what we have in our brain is like a simulation engine, if we do something we can think to what would happen.

![[Pasted image 20230506173450.png]]

## Scale Improves Robustness

![[Pasted image 20230506173615.png]]

As we scale (MNIST), we can saw that is a jump in robustness, so after increasing size after certain point we find this good spot.

It's showed that Scale is a low of robustness, is we fix a "reasonable" function, for example sigmoid. (Smooth function), a parametrisized one with as many parameters, so a neural net, but not a [[Kolmogrov-Arnold net]] (These are non smooth).

Sampling $n$ data from **truly high dimensional** distribution (mixture of Gaussians) in MNIST we have (60k datapoints, 28x28x1 = 784), instead ImageNet is 256x256x3. And add label noise, to make problem harder to our Nets. 

Now what we do is to **memorize** the dataset such by obtaining training error equal to $0$

$$p \gt\gt nd$$

The hard part is after we reached 97%, that is hard because we have noise in labels, the hard part is below noise level, and to do this robustly, (being Lipchitz).

Where [[Lipschitzness]] defined as 

```ad-hint
Changing the input doesn't dramatically change the output of our function`
```

If input is 784, and 60k datapoints, minimum size of MNIST dataset has to be 10^8, so we need 10^9 neural network to robustly learn MNIST.

## Why dramatic Overapametrization

Intuitively memorizing $n$ datapoints is about satisfying $n$ equations, so order of $n$ parameters should be enough.

```ad-important
**Theorem (Baum 1988)**
Showed two layer neural net with threshold activation only need $p \approx O(n)$ to memorize binary labels
```

This is showed for ReLU and [[Neural Tangent Kernels]]: process of training with GD is a [[Dynamical and Complex Systems]], you update weights as you go further, and these learning dynamics are modelled by [[Differential Equations 1]] on how updates of gradient descent would work. Increasing size of neural net to infinite width the dynamic of learning has a closed form solution that is a [[Kernel Method]].

### Real example
![[Pasted image 20230506180432.png]]

The effective dimension should be accounted because different images are us the principle components of an image are important, information is much smaller with respect to data, is difficult to say what is the effective dimension.

And with ImageNet the $n \approx 10^7$ $d=10^5$ with $d_{eff} \approx 10^3$ so we need $10^{10}$ parameters to learn it.

Now comin back to Overparametrization...

![[Pasted image 20230506180815.png]]

We would like to escape from that regime, but how can we do? [[Neuroscience]]!

## Liquid Neural Networks

Another time we inspire to biological brains. In order to break the law of robustness.
![[Pasted image 20230506180914.png]]

These are called [[Liquid Neural Networks]].

We start by analyzing [[Nervous Systems]] and [[Neural Circuits]]

What are the building blocks of these? 

- Interaction between neurons and this is a continuous time process, neural dynamics are continuous processed described by differential equations.
- Synaptic release: connecting nodes in neural network in modeled by scalar rates, but we could model it as an entire probability distribution. "How many neurotransmitter generated from a cell interact with another (the channels) and how it can activate". Communication between only two nodes is sophisticated, a fundamental building block for our intelligence.
- We have parallelization, recurrence, memory and sparsity.

So now what can we do with this? 

Incorporating this we get much better [[Representation Learning]] build on top of continuous time process

