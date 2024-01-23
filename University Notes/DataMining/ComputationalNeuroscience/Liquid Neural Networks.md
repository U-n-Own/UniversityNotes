Related stuff : [[Neuroscience]], [[Neuromorphic Computing]], [[Nervous Systems]].

Discovered in : [[Moder Era of Statistics]]

Basic equation of a Liquid Networks come from the interaction of two neuron cells

$$
\frac{d\textbf{x}(t)}{dt} = \frac{-\textbf{x}(t)}{\tau}+\textbf{S}(t)
$$
$$
\textbf{S}(t)=f(\textbf{x}(t),\textbf{I}(t),t,\theta)(A-\textbf{x}(t))
$$

![[Pasted image 20230508123153.png]]

We can get some properties like *Causality*, *Expressivity*, *Memory*, *Robustness*, and *Extrapolation*: going beyond data we train on, out of distribution, one example is robotics real word with MixedHorizon decision making
![[Pasted image 20230508123251.png]]

As we discussed in [[Moder Era of Statistics#Liquid Neural Networks]]

![[Pasted image 20230508124153.png]]

## Continuous time NN

![[Pasted image 20230508124435.png]]

This neural net is an $f$ having layers, width, and certain activation, receive recurrent connection from other cells and **exogenous input** the $I(t)$, and is parameterized with a $\theta$.

This network is parameterizing the *derivative of the hidden states* and not the hidden state itself, this is a continuous network. So the update (output of our network) is generating updates for the derivative of the hidden states, this is a dynamical system.

### Little excursus on Residual Networks

[[Residual Networks]] are basically network that uses **Skip connections** from a layer to anothe by propagating information without actually doing anything, the skip connection is modelled with that equation, and each black dot is a computation that was done in the system. 

Computation happens at each layer.

In continous process computation can be done in arbitralily point in a vector field. (Adaptive computation).

