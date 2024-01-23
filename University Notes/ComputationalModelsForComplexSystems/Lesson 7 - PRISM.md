Date: @today

Status: #notes

Tags: [[Dynamical and Complex Systems]]

# A new logic CTL

CTL, *computational tree logic*, keeps traces in a tree like structure we can see how the grammar is built

State formulae:

$$
\Phi ::= true|a|\Phi\wedge\Phi|\neg\Phi|\forall\Psi|\exists\Psi
$$

Path formulae: where F stands for future, G for global (all reachable state), U stands for until, we want that formula the first be satisfied until there is a fail.

$$
\Psi ::= X\Phi|F\Phi|G\Phi|\Phi U\Phi
$$

## Probabilistic CTL

Where $p \in [0,1]$ is a probability bound

$$
\Phi ::= true|a|\Phi\wedge\Phi|\neg\Phi|P_{\approx p}[\Psi]
$$

Here the second one is a "bounded untiil", while the second remains until.

$$
\Psi ::= X\Phi|\Phi U^{\leq k}\Phi|\Phi U\Phi
$$


## CTL Sematics

![[Pasted image 20230330122149.png]]
>[!info]
> 

## Lotka Reaction in Prism

This model has an infinite number of states


# Inferring parameters from a simulation

For certain chemical reaction or infection, we can't have the data just yet, but we can infer it.
[Paper from professor](file:///home/vincentkun/Downloads/DataMod_2020_PRISM_Covid19.pdf)

![[Pasted image 20230331150412.png]]

We're doing some assumption to simplify the model. SIR model do some simplification, for example we don't know the number of people with the infection on going but with no symptoms, we can infer this maybe?

Starting by modifying SIR model

![[Pasted image 20230331150816.png]]

## A new SIR model

Our $p(t)$ is a function of time and is a prevention mechanism, and it's implemented as a piecewise linear function
![[Pasted image 20230331152416.png]]

$p_{lock}$ is a number in $[0,1]$ that determines if we are in lock state. $N$ is the day which lockdown is observed.
![[Pasted image 20230331152536.png]]
Code used is this

Where prev is our $p_{lock}$, using odeint to computate our set of differential equation solutions.
![[Pasted image 20230331153100.png]]

---
# References

