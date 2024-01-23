[[Machine Learning]],[[EMERGE]],[[Recurrent Neural Network]], [[Dynamical System]],[[Dynamical Systems]],[[ECML]]

![[Pasted image 20230919174800.png]]

We want something that process information adaptively, so Neural Networks!

## Desiderable properties

Model dynamic data, the world is dynamical and we need something that can model evolution trought time!
Also we need to control this system, we want guarantees that these won't have out of range responses


![[Pasted image 20230919175025.png]]

What dynamical system? 

## Oscillators as dynamical system

These are quite suitable and simple, remember we have to model energy and dissipation of it.

$$ \ddot{y} = -\gamma y - \epsilon\dot{y} + f(t)$$
Spring and mass collect potential energy and kinetic, mass we assume constant 1. That is the differential equation describing accelleration of mass in space.

First component is oscillatory motion around equilibrium point without energy dissipation : periodicity
Second is a dampening term that dissipate the energy, so stops oscillating : [[Fading memory]]

The last term in an external force we give to the system, the input stimuli in the neural network, this is a sound physical system, how do this describe a neural net? 

## Multistable Oscillators Networks

A network of dampened harmonic oscillators!
Capable of extracting from input different characterizing signals

Driven by a non-linearity, what?

$$f(t) = tanh(Wy+b)$$
By means of having an eterogeneous selection of **Stiffness** and **Dampening** factors!

Then we couple them, and they becomes a sort of RNN. This make a *multistable system* by adding (our force input) that non-linearity: It has stable and unstable states, for remembering initial conditions for example, we want a model able to express several range of our problem complexity.

Now if we put this all togheter in a graph with fancy colors we get this

![[Pasted image 20230919181958.png]]

These oscillators are excitated by these blu and red stuff, we call it **Excitation layer**, which is really similar to a neural layer, provide the exitatio to oscillator that give us state of the system. 

Then what we do, we do a substitution of variable, we subtitute the $y$ velocity inside the $tanh$, and there we use Euler discretization.

Now this is describing evolution in time discretized, of our neural network, a network consistently behavioring like oscillators, multistable coupled oscillators.

![[Pasted image 20230919182221.png]]

## Adding reservoir (of course)

All the parameters inside the cyan are untrained, we train only the readout, reading the oscillators output and outputting a value, as we do in reservoir computing.
Then we treat as **hyperparamters** $\gamma$ and $\epsilon$ the dampening and stiffness: random frequency interval, random dampening intervall

Of course $W,V,b$ are fixed and initialized as in Echo-state Networks


![[Pasted image 20230919182444.png]]

## Linear stability of RCO

How do we choose the hyperparameters? Linear stability analysis!

We can derive the sufficient condition for linear stability, and discover that are beautiful but too restrictive leading to poor performance, the space to parameters that satisfy is zero. Then you go trough the math and derive necessary conditions and discover intresting necessary conditions!

![[Pasted image 20230919183312.png]]

And the trick is: The eigenvalues of the jacobian of these systems lie in this union of these discs in the unitary circle, so we push the discs inside it, we find the conditions for which these are all inside it. 

What we get is a very rich spectral diversity when set up the network.

We have different $\epsilon,\gamma$ for each neuron, so basically we pick up points in space for the two and then we sample in some circle sourrounding it and give these values to oscillators!

![[Pasted image 20230919183532.png]]

Then we get the experimental part

## Experiments

![[Pasted image 20230919183835.png]]
### Regression

![[Pasted image 20230919184000.png]]

### Results on necessary conditions
![[Pasted image 20230919184107.png]]


## Conclusions


We're searching for a means to use dynamical systems to characterize and execute neural nets these can be competitive even untrained, these are able to describe unstable chaotic and stable behaviour which is nice.
#### Future works

Implement nets in physical systems.

How can we be sure that this model is the best to represent this kind of behaviours? They use fully randomize approach, we can adapt the system to the domain of input signal withot using backprop, so like pretraining, continual learning, making sure that system stays in critical state. 

We could compose these networks of networks, what happens? Properties? Does it break? 

![[Pasted image 20230919184928.png]]
![[Pasted image 20230919184947.png]]

With this vocabulary you could describe neural processing but also neural processing of swarm of robots. This is the EMERGE project.
