Date: [[2023-03-22]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Boltzmann machine are Conditional Random Field

![[Pasted image 20230330210021.png]]


Boltzmann machine are MRF defined over binary RV. Since is a MRF we have an energy function a linear combination of our RV, there are coupling between RV (correlation between them) $M_{ij}S_{i}S_{j}$
![[Pasted image 20230330205553.png]]
A structural property is absence of self recurrence, no selfloop in this case. The parameters of the model are those of the energy function ($M,b$).
Boltzmann machines are a type of RNNs the $b_j$ becomes a bias term and the parameter encode interactions bewteen variables (observable and not). But these haven't *recurrent* connectivities.
The network activity is a sample from *posterior probability given inputs*: visible data.

Image of Restricted Boltmann Machine 

![[Pasted image 20230330205917.png]]


## Stochastic Network and Boltzmann Machines

In Stochastic neural network we get every time a different input because a "neuron" that is more active is more likely to output a 1 and less actives outputs 0, so we're sampling from these networks, instead standard Neural Nets are deterministic.

This is a MRF a Neural Network in details a stochastic neural network! 
We want a probability of spiking neurons.

## Stochastic Binary Neurons

Spiking point neuron with binary outputs $s_{j} \in \{0,1\}$. Typically discrete time model with time in small $\Delta t$ intervals.
At each time we want a probability that a neuron *emits a spike*

$$
s_j^{(t)}=
\begin{cases} 
1, \;\; p_j^{(t)} \\
0, \;\;1-p_j^{(t)} \\
\end{cases}
$$

![[Pasted image 20230401182835.png]]

## General Sigmoidal Stochastic Binary Nets


We have a
- Weight matrix $M$ = $[M_{ij}]_{i,j}$, $\in \{1..N\}$
- Bias vector $b$ : $b_{j}\in \{1...N\}$

Local neuron potential $x_j$ is the $net$ how we saw at Machine Learning course.
![[Pasted image 20230401182933.png]]
A chosen neuron fire with spiking probability
![[Pasted image 20230401183016.png]]

## Parallel Dynamics

How does the model state (activation of all neurons) evolve in time?

Assuming RV are update at evey timestep...

![[Pasted image 20230401183055.png]]

We have a **Markov Process** as we can see, introducing by marginalization the previous states as we did with Markov Models. This is how we use to update boltzmann machines (update in parallel). Update one at a time is good for theory, all togheter is good for computation.

![[Pasted image 20230401183110.png]]

## Glauber Dynamics


This is a dynamical system that can converge or diverge one neuron at each step it's chosen to fire or not, it has stationary distribution, and if we know that distribution we know how to train (knowing global joint distribution). If there is a stationary distribution...then the system is said to be ergodic, meaning that it will eventually visit every possible state with a certain probability. In such a system, we can use Markov Chain Monte Carlo (MCMC) methods to sample from the stationary distribution and thereby estimate the global joint distribution of the system.

Knowing the global joint distribution allows us to train the system by optimizing a loss function that measures how well the system output matches the desired output. This can be done using techniques such as maximum likelihood estimation or stochastic gradient descent. Maximum likelihood estimation involves finding the parameters of the model that maximize the probability of observing the training data, given the model. Stochastic gradient descent, on the other hand, involves iteratively updating the model parameters using gradients of the loss function with respect to those parameters.

Once the system is trained, it can be used to make predictions on new input data by computing its output distribution given the input. In some cases, this output distribution can be used directly as a prediction. In
![[Pasted image 20230401183356.png]]

## Boltzmann-Gibbs Distribution

This is a property of mechanical statistics we need this because describes that we can have reversible transition guaranteeing existence of equilibrium. To train these models we need to take sample, for example Gibbs sampling, we need to build markov chain that converge with the distribution that we want to approximate.

*Undirected connectivity enforces detail balanced distribution*:

$$
P(s)T(s'|s) = P(s')T(s|s')
$$


This is the classical form of the stationary distribution of the RV (distribution of activation of neurons)

$$
P_\infty(s) = \frac{e^{-E(s)}}{Z}
$$

Where $E(s)$ is the energy function and $Z$ is the partition function we saw in [[Lesson 8 - Markov Random Fields#MFRs]]

## Learning

We now desider to derive equations, we will approximate by sampling and we want to build a sampling process

```ad-cite
$\color{red}{\text{[Ackley,Hinton and Sejnonwsky]}}$

Theorem: Boltzmann machines can be trained so that equilibrium distribution tends towards any arbitrary distribution across binary vectors given samples from that distribution
```

Simplification:
- Bias $b$ absorbed into the weight matrix $M$
- Consider only visible RV $s=v$
Use probabilistic learning thecniques to fit parameters like `maximizing the log-likelihood`
![[Pasted image 20230401184831.png]]
We take the  $\frac{\partial\mathcal{L}(M)}{\partial M_{ij}}$ = $-<v_i,v_j>+v_iv_j$

### Derivation

$P(V^{l}| M) = \frac{exp(-E(v^l))}{Z}$  $\Leftarrow$ Definition of

$log P(V^{l}|M)= -E(V^{l)}-logZ$

- where $v^{T}Mv=\sum\limits_{p,q}M_{pq}v_q,v_{p}$ when we derive all becomes zero but the $i,j$

$\frac{\partial log(P(V^{l}| M))}{\partial M_{ij}} = \frac{v_{i}v_{j}-\sum\limits exp^{-E}}{Z\times v_{i},v_{j}}$
.
.
.
$\frac{\partial log(P(V^{l}| M))}{\partial M_{ij}} = v_i,v_{j}- \sum\limits_{v}P(U|M)v_{i}v_{j}$
but this is almost $\mathbb{E}[v_i,v_j]_{v\approx P(V|M)}$. This is called this way $<v_i,v_j>$

that is = $v_i^l,v_{j}^l-<v_i,v_j>$

$\frac{1}{L}\sum\limits_l$


when $\frac{\partial\sum\limits log p}{\partial M_{ij}} = <v_{i},v_{j}>_{D}-<v_i,v_{j}>_{M}=0$

Remember that $v_{i},v_{j}$ are really simple just 0 or 1.

#### Hebbian Learning a neural interpretation

$M'_{ij}=M_{ij}+\alpha\Delta M_ij$

this thing we derived by only probabilistic formulation is a mechanism that is taken from the brain, basically Hebbian says the correlation of firing neuron and Anti Hebbian tells the anticorrelation. So neurons that fire togheter are connected togheter and stay like that, instead neurons that fire but not in the neighbour are disconected between them. This is the Hebbian and Anti-Hebbian. Certain time it works like regularizer the fact that we use antihebbian

We have *wake* part that is standard hebbian rule applied to empirical distribution of data that machine sees coing in from the outside world.

---
Note: This part was taken using GPT-3.5 Turbo

*Dream* part instead in hebbian learning is a theoretical concept in neuroscience that suggests that during sleep, the brain consolidates and strengthens memories formed during waking hours through a process of synaptic plasticity known as Hebbian learning. This process involves the strengthening of connections between neurons that are activated together, leading to the formation of neural networks that represent specific memories or skills. According to this theory, dreaming may play a crucial role in this process by allowing the brain to reactivate and strengthen these neural networks during sleep, thereby facilitating memory consolidation and learning.

The *Wake* part in hebbian learning instead can be seen as the process of strengthening or weakening the connections between neurons based on their co-activation. This is achieved through the repeated activation of a specific neural pathway, which leads to the reinforcement of the connections between the neurons involved in that pathway. The Wake part in Hebbian learning is therefore essential for the formation and consolidation of neural circuits that underlie learning and memory processes. It enables the brain to adapt to new experiences and stimuli by modifying its neural connections, which allows for more efficient processing of information. 

---

##### On Hebbian Learning

Intrestingly hebbian learning can be found when thinking about recurrent neural networks structures like ***Hopefield nets***, that stores "memories".
Remember that weights are our synapses and why multiplications? 
![[Pasted image 20230330182307.png]]

Negative when neurons disagree positive when neurons agree! Positive connections let keep states, negative wants to change it.
![[Pasted image 20230330182405.png]]

#### Example of memories

Some initial states: randomly initialized
![[Pasted image 20230412173701.png]]

Final states of our network. Sometime can "remember" mixture of patterns in memory.
![[Pasted image 20230412173740.png]]


## Learning with hidden variable

Well re deriving all log lieklihood with respect to $M_{ij}$ the correspond summation is like that we've seen before. Now the clumped those with supescript $c$ expectation: $<s_i,s_j>_{c}-<s_i,s_j>$ these clumpd are the first term (those that sums over $h$ marginalizing over hidden) and they are multiplied by the posterior $P(h|v)$. Those term are maginalizing out a lot of stuff and are difficult to compute. Marginalizing the hidden and pluggin in the visible in first term and maginalizing on all the $s$ RV both visible and hidden. Those two expectation are difficult we approximate them by **sampling** we start pooling out samples, and them multipling and averaging them and getting an estimate that is $<s_i,s_j>_c$ (sampling only hidden given the other we know) then from another sampling process in which we sample from all the variables.

![[Pasted image 20230401190706.png]]

## Restricted Boltzmann Machines (RBM)

Those we talked until now is a nightmare so we restrict this to a thing that is very similar to neural nets
![[Pasted image 20230401190731.png]]
But since connection are bipartite graph we have a nice RNN!

What we learn in this case? Things are tractable now because we can work in parallel and speedup there is less coupling and is easier to sample

### RBM Catch

RBMs hidden units are conditionally independent given visible units, and viceversa.

$$
P(h_{i}| v) = \sigma(\sum\limits_{i}M_{ij}v_i+c_{j})
$$


### Training Restricted Boltzmann Machines

Again by likelihood maximization we have 

$$
\frac{\partial\mathcal{L}}{\partial M_{ij}} = <v_ih_j>_c - <v_ih_j>_{model}

$$

By gibbs sampling approach what we do is the following:

1. We start with an initial sample from the distribution we want to simulate.

2. We then update each component of this sample in turn, using conditional distributions that depend only on the other components.

3. We repeat step 2 many times, updating each component in a cyclic or random order, until the updated sample converges to the desired distribution.

The advantage of this approach is that it can be used to simulate from high-dimensional distributions where direct sampling is not feasible or efficient. 

- Wake
	-  Clamp data on $v$ 
		- **On Clamping**:  Well once you clamp data on visible units the equilibrium distribution of hidden units can be computed in one step because are independent one from another given the states of visible units. But in what consists clamping? So clamping refers to fixing the values of some visible units to a specific value, rather than letting them vary according to their probability distribution. This is typically done in the beginning of training a deep belief network, where the visible units are clamped to the training data, so that the network can learn to model the underlying probability distribution of the data. Clamping can also be used during inference, where we fix some observed variables and infer the values of other variables.
		  The important part is : *once you clamp data vector $v$* you can reach **thermal equilibrium** in one step!
	- Sample $v_ih_j$ for all pair of connected units where $h_j$ are computed by $P(h_j | v)$ given the visible ones.
	- Repeat for all elements of dataset

- Dream 
	- Don't clamp units starting with random assignments
	- Let network reach equilibrium

In principle you get at the start your data samples, and then wait to infinity to get the expectation, we cannot wait infinity so we trucate, hoping that these have not high variance (because). We cut at the *frist step*. Does it work?


## What does constrastive divergences learn?

Well this is a very crude approximation of the gradient of the log-likelihood but it doesn't follow the gradient and even isn't maximizing or minimizing.

why use it? Because Hinto says it.



Code seen in slides: 

First we have the *wake* part.


For the *dream* part we need the $v$ terms computed before, we compute the stochastic activation of $h_0$ and having that we compute $P(v^1|h_0)$ we can draw from probability (reconData). And then use those to compute $P(h_1|v^1)$, then you can sample the activation of neurons.


#### Boltzmann machines in python

There is some implementation is scikit.learn and othe stuff...


## RBM for netflix competition

By Geoffrey Hinton lesson:

![[Pasted image 20230402160638.png]]

## Character recognition

Learning good features for reconsturcting images of number 2 handwriting: MNIST data

In mnist data there is a visible unit for each pixel, we're representing in the hidden space the pixel using 50 binary neurons. So we project 16x16 binary image into 50 binary feature neurons. So each unit is connected to all the visible unit. So each hidden neuron will learn a matrix that is the feature that neuron responds to.

-- show weights


## Final reason for intorducing RBM

This is an autoencoder build by pretraining each layer using RBM and this what created the Deep Learning we know now





### Take home lesson

Boltzmann machine a neural network that's not deterministic, less used in practice, more theoretical. These makes sense to know them.
Next time we will introduce Deep Learning, introducing CNN, breaking it down in decomposing pieces to understand and putting in togheter in much complex model architectures to understand interesting mechanism.

```ad-summary


```


---
# References

