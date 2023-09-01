Date: [[2023-05-02]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

Links : [[Reservoir Computing]] [[Recurrent Neural Network]], [[Neuroscience]], [[ ComputationalNeuroscience]],

# An Extremely efficeint way of designing and training RNNs

We spoke about RNNs extensively in [[Lesson 14 - RNN Sequential Models and Gated RNN]], a well known characteristic is *weight sharing*. The forward computation present two problems: explosion and vanishing gradient issues. This problem is still unsolved. We will se a solution.

From vanilla RNNs we saw gated RNNs like LSTMs and GRUs, alleviating the gradient issues.

Philosophy here is: 

```ad-quote
Randomization is cheaper than optimization
```

When price is energy and big models: there is a paper called Green AI.

Merge togheter how brain working and how computer works this is called :[[Neuromorphic Computing]]

What they do is to focus on architectural biases, so accuracy and complexity matters, linear models are low in complexity but low accuracy, then SVM and deep NNs are more complex but more accurate, Randomized NNs instead are not that complex but are accurate like the most big ones.

## Randomized Recurrent Neural Networks

Seen it in course of ML, basically keep randomized weights and train the last *readout layer*.

This is *amaneable* to some neuromorphic implementation, you find parameter and fix it.


## Reservoir Computing 

We focus on the dynamical system, we need to achieve some stability conditions, and *Echo state property* will grant this stable, what we do is randomize and scale the matrix.

- Large layer of recurrent unis
- Sparse connections
- Random initialization
- Untrained

Training is really simple

$$y_t = h_tW_{hy}$$
Can be trained in closed form, we're just doing forward pass and compute readout.

$$W_{hy} = (H^TH)^{-1}H^TD$$
The matrix $H$ is big but not that big because we're taking the dimension of the **Reservoir** that is an hyperparameter

Then we use Cover's Theorem by embedding in higher dimensional space to get a linar solution.

Some more notation:

Reservoir = Discrete-time input-driven dyn. system.

[insert slide here]

We're defining the vanilla RNNs by iteration using a transition function $\hat{F}(s,h_0)$

### Echo state property

Stableness property: 

[insert slide]


If we iterate the reservoir equation where $h_0$ is final state and $z_0$ converge to zero, so start with two initial condition that start not equal and end up being more similar. We want a fading memory behaviour but not too much.

[insert slide]

#### Condition for ESP

Necessarily condition: the spectral radious of $W_{hh}$ is involved

Sufficient: [insert slide]

Recurrent matrix has a singular value if is less than 1 is contracts and will converge to zero: $$||W_{hh}||_2 \lt 1$$
This is a strong condition we want it t be contractive. (Too much fading memory isn't good)

The necessary one instead if the spectral radious is smaller than 1. If bigger won't converge but this leads to vanishing.

## Reservoir initialization

Generating a random matrix $W$ from $\mathcal{U}$ distribution on $[-1,1]$.
Then scale by desired spectral radious
$$insert equation$$
[Insert slide]

input scaling is the most important hyperparameter of reservoir

## Dynamical Transient

[insert slide]

The *washout* is important for regression task, we have to wait for syncronization of reservoir to happen.
We want ***Stable Dynamic***, we've seen it in [[Dynamical and Complex Systems]] course
[inert slide]


[insert slide] esn training

linear regression is key

[readout training slide]

[practica tips]


## Architectural variant

Change readouts in accord to the tasks

[insert slide]

*Feedback connections* from readout gives us a reservoir that isn't randomized anymore but we have a training involved in this case, introducing an hyperparameter $\alpha$ that if we move in the space of


## Application of RC

Nets that learn behaviour of some gramdma wondering in an enviroment

![[Pasted image 20230502121804.png]]


![[Pasted image 20230502122035.png]]

![[Pasted image 20230502122049.png]]
![[Pasted image 20230502122322.png]]

![[Pasted image 20230502122342.png]]

![[Pasted image 20230502122631.png]]

## Cutting edge RC

We want a stable reservoir, and we analyze our dynamical systems.
We not an unstable but not only stabilized, we don't want to step or stay too much into chaos.

![[Pasted image 20230502122647.png]]

Think of the $f$ as tanh, we want to train neurons in unsupervised manner, for smaller $a$ we get linear

![[Pasted image 20230502122840.png]]

A measure used in dynamical system to 

If greater than 0 expands and we have an exponential sensitivity to initial condition
![[Pasted image 20230502123041.png]]


We've an orthogonal matrix that describe the topology of this reservoir

![[Pasted image 20230502123207.png]]

## Universal approximation theoresm for ESN

Proved for non-linear and linear readout, as we can see non-linearity is still requested

![[Pasted image 20230502123307.png]]

SOTA for neuromorphic computing is in the next paper : Photonics 

![[Pasted image 20230502123426.png]]

## Deep Echo State Networks

Using activation of reservoirs for more reservoirs and then use hidden activations, or just use the last one for the readout. A deep reservoir is a nested set of dynamical systems

![[Pasted image 20230502123502.png]]

![[Pasted image 20230502123612.png]]

## Architectural bias of RC

We've stability but more sensitive

![[Pasted image 20230502123644.png]]

## Memory capacity

![[Pasted image 20230502123729.png]]

## Edge of Chaos measure

As we can see we approach the zero (edge of chaos)

![[Pasted image 20230502123805.png]]

## Neural network for graph structure

Time steps are vertexes, for each vertex we have and embedding as embedding of the neighbors + input features

![[Pasted image 20230502123938.png]]

This $H$ is a dynamical system that converge to a fixpoint.
Use this fixpoint as input for another reservoir and so on, until we get to the last layer, this is a collection of features and we can do classification. Reaching SOTA performance
![[Pasted image 20230502124047.png]]

## Eular state networks

Antisymmetric matrix has an eigenspectrum.

For discret time dynamical system we have a different thing

$W_h - W_h^T$ is the trick to obtian 

![[Pasted image 20230502124328.png]]

![[Pasted image 20230502124525.png]]

## Edge of stability ESN

What is the architectural bias here?
Pushing alpha to zero eigenspectrum collapse when when pushing to beta to zero you observer convergence towards unitary

![[Pasted image 20230502124705.png]]
What if we train this?
Results of RoaRNN, well this is a solution to the
![[Pasted image 20230502124946.png]]

### Take home lesson

```ad-summary


```


---
# References

