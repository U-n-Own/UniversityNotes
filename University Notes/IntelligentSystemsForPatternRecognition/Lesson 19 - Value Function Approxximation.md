Date: [[2023-04-20]]

Status: #notes

Tags: #ispr

# Reinforcement Learning on the Scale

Until now we've seen that our $v$ and $q$ are vector and matrixes, we can't work all the time with tabular enviroment, state space is finite, but sometime on big or infinite state spaces for example Molecule search : $\gt 10^{60}$

Also tabular representation are bad for generalization, these are basically lookup table, and with Curse of dimensionality becomes very difficult, these are very sample inefficient, for example if we're in a grid being in state $a$ or $b$ is radically different.

## Value functon approximation approaches


![[Pasted image 20230420150303.png]]

Which function approximation?

Linear, Neural netowkr, Decision tree, Nearest Neighbour, fourier or wavelet bases, we will focus on differentiable methods so linear combinations of features or neural network. 

### Stochastic Gradient Descent

Instantaneous error with respect to the sample (episode montecarlo talking), the error is $v_\pi$ the groud truth, we don't have it, we can use monte carlo or other proxy

![[Pasted image 20230420150540.png]]

### Feature Vector states

![[Pasted image 20230420150756.png]]

Neural embedding compress something from high dimensional spaces to small.

## Incremental predictions

Since we don't have $v_\pi$ we can replace it for an estimator for this ground truth, and we know in MC the estimate is the *return*. High variance but low bias.

If we want a low variance we can use TD(0), where we use the current return and finally in $TD(\lambda)$ we use the eligibility trace. This time we're moving in parameter state moving by direction indicated by gradient using difference by pseudo target and estimation.

### In montecarlo

What we learn is supervised learning on couple of $S_i, G_i$ (samples) which we get an associated return. But montecarlo has high variance, but also good properties, we know that converges to local optima in few steps, with this we have the security to find convergence to local optima. 

![[Pasted image 20230420151235.png]]

### In TD

![[Pasted image 20230420152154.png]]

### In TD($\lambda$)

Assigning credits on frequency and recency euristic again, accumulating these metric for each dimension of the parameters.
Assigning more responsabilities to parameters that have higher derivative, and higher derivative is the recent thing.

![[Pasted image 20230420152551.png]]

## Incremental Control

Nesting policy evaluation and optimization

![[Pasted image 20230420152655.png]]

## Convergence of prediction
In table lookup all is good all converge and reach optima, in all the ways. As soon we start approximating stuff, things are going bad, but things are good with monte carlo because converges!
Both off policy than on policy, in bootstrap is when thing gets worse, for TD and both linear-non linear.
![[Pasted image 20230420152958.png]]

Being fully stochastic make model learning really difficult, because in that way our samples that are Non-IID, "traces", these traces are dependant, so we need to work in another way, using batch methods.

## Batch methods

We're using the same data usually with SGD, but we don't need to repeate experience, we just want to store and collect these memories.

![[Pasted image 20230420153537.png]]

Performing learning by sampling in the dataset (a buffer), before we were sampling from policy.

![[Pasted image 20230420153552.png]]

## Deep Q-Networks

The model trained to playing atari games was trained with this model, what we do is take an acation and put into the buffer, what happen when the agent play is to store a lot of actions and the we sample from the buffer.


![[Pasted image 20230420153956.png]]

What are the $w_i^-$ and $w_i$? In the context of Q-letwork and Q-learning target our loss is and expectation where the two terms $w_i^-$ and $w_i$ are used to weight the importance of each sample 

## ATARI-DQN

![[Pasted image 20230420154340.png]]

![[Pasted image 20230420154628.png]]

### Ablation study
Using replay buffer really is better from 3.17 to 240.73 as cumulative reward.

![[Pasted image 20230420154637.png]]

## Improvements of DQN

We use the two $w$ the one with minus to stabilize our Network.

If we're learning Q-function we're learning the V term and the A term, how good is to be in a state and how good is to be in a state with an action, so we train the same network one with A and other with V then we sum up to our Q.

There are some states where we have to differentiate between the action is really important.


### Take home lesson

We've moved from tabular environment in an approximated word, losing convergence guarantees especially off policy with bootstrapping and model free such with atari by crafting thing appropiately, sampling from buffer, using fixed target, dual networks to stabilize target and use trick like advatage function to improve efficiency.

Next week we will see policy gradients where we parametrize the policy instead of value function.

```ad-summary


```


---
# References

