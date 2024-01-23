Date: @today

Status: #notes

Tags: [[Dynamical and Complex Systems]], [[A.I. Master Degree @Unipi]], [[Probability]]

# Simulating stochastic reactions

Why introduce stochastic approach? Since differential equation are **deterministics**, we cannot really describe accurately these chemical reactions. We have to define **stochastic simulation algorithms**.

## Predicting reaction

It's difficult to predict in advance some complex reaction, so what molecule will react next, the choice of the next reaction type and so on.

Moreover having a large number of molecules, we don't bother because of large number law. But working on small number of molecules is another thing, randomness is crucial.

## Gillespie’s Stochastic Approach

Gillespie's Stochstic Simulation Algorithm (SSA), at each iteration take into account the randomness of the quantities in the reaction, assuming a reaction constant $c_\mu$ for each reaction

Given a set of $\mathcal{R}$ reactions

We have $c_{\mu}dt$ as probability that particular combination of reactants react in an infinitesimal $dt$.

The constant $c_{\mu}$ is used to compute the rate of reactions and is a values dependent to the whole reaction $a_\mu$

$$
a_{\mu}= h_{\mu}c_\mu
$$

where $h_{\mu}$ is the number of distinct reactant combinations.




![[Pasted image 20230316113108.png]]


Note that the last time in non stochastic environment the $k_1[S_1][S_{2}]\approx a_1$ for $k_{1}= c_1$, so with a slight approximation we get basically the same results.

This assumption is an empyrical results as we said last time

## The exponential Distribution role

Now for each reaction we have a **propensity** $a_\mu$, and we can use this as parameter of an exponetial distribution, basically analogous of geometrical distribution in continuous setting. 

The PDF and CDF are:

![[Pasted image 20230316113819.png]]

The role of the lambda in this case is the frequency of our events, the mean is $\frac{1}{\lambda}$, so a frequency of $2$, denotes a time of $0.5$ time between each event.

### Exponential distribution definition [informal]

Informally an exponential distribution is a Poisson process in a time from 0 to $\infty$ that describes in which events happens independently and continuously at a certain constant average rate. $\frac{1}{\lambda}$

![[Pasted image 20230316113902.png]]


We know that exponential distribution is **Memoryless** so is very good for represent stochastic processes!

$$
P(X > t + s | X > s) = P(X > t)
$$

So if we model two reaction we don't mind about the past, we say usually **forget about the history** of the simulation.

If we have a set of $X_{1,}...X_{n}$ inependent exponentially distributed we don't have to keep the old values. Let $\lambda_1,...,\lambda_n$ be the parameters then

$$
X = min(X_{1},...X_{n}) \;\; \text{is exponentially distributed!}
$$

with parameter: $\lambda = \sum\limits\lambda_{1}, ..., \lambda_{n}$


## Gillespie's Stochastic Algorithm (SSA)


We generate an unique variable representing a vector of distribution, and a $t$, the algorithm iterates basically at each time step and choose randomly an $a_0$,
 then to choose what will be the reaction will randomly sample from those with the probability in step below, basically we are choosing in a probability distribution weighted by the propensities.

![[Pasted image 20230316114624.png]]

## Example

![[Pasted image 20230316115000.png]]

We have two steps that uses probabilities distribution, one uses an exponential (step 1), step 2 instead sample

## Implementation details about SSA

![[Pasted image 20230316115439.png]]


Then we get the inverse of $F(x)$

![[Pasted image 20230316115825.png]]


![[Pasted image 20230316115937.png]]

We want to get that $\mu$ that is the smallest integer $k$ that satisties the sum of the $a_i$ greater than $na_0$

# Visualizing ODEs and SSA

Enzymatic activity:

![[Pasted image 20230316120109.png]]


The discretization done by SSA is such that we have only one reaction picked at a time step, in a probabilistic way, instead in ODEs if more reaction can take place, all of them will take place, this is what make it smoother.


## Lotka-Volterra

Reminding last time we saw how with ODEs [[Lesson 2 - Continuous Dynamicals Models#Lotka-Volterra equations]], they were steady in some cases, or cyclic. Instead what we see now with SSA is this.
![[Pasted image 20230316120707.png]]

## Negative feedback-loop

Let's talk about genes: $G_{1},G_{2},G_{3}$ produce the proteins $P_{1}, P_{2}, P_{3}$, and let's says that proteins inhibite the genes like this $P_1$ binds to $G_2$ this molecule cannot produce $P_2$. And the other like this, with $P_2$ disappearing faster, 100 time faster.

Starting with $G_{i}=1$ and $P_i$ = 0, now we will compare ODEs with SSA

![[Pasted image 20230316121227.png]]

ODEs comes to a steady state, instead SSA doesn't get to a steady state, we can see that quantity of $P_2$ decrease and increase very fast, spiking. The other also varies in a stochastic way.

## Computational cost of SSA

So which of the two approach should i use? 

Stochastic approach is more accurate, describe the reactione one by one, what if we want use every time computational approach?
Well... computational cost is expensive for large number of molecules. But if we have large number of molecules we can approximate the stochasticity with the law of large numbers. This becomes acceptable. 

So as general guideline if you want to describe a system with a lot of components of the same type, the ODEs are much more effective, instead with very few components or slow reaction for which the time is essential, stochastic approach is more likely to be used also if this is more costly.

### Variants of SSA (exact approaches)

- Reducing complexity to choice of reaction using efficient data structures like binary trees by Gibson and Bruck
- Decreasing propensity using a list with the knowledge of the reaction with the high probability.

### Vairants of SSA (approximate approaches)

- **$\tau$-leaping methods**: the idea is instead to execute reactions one by one, you could repeat the same reaction $n$ times, assuming that the likelyhood to appear stay constant more or less, we use a threashold to get the error not too high, in many cases we can't even see the difference with standard approach if the assumption holds, in many cases this can be useful.
- **ssSSA** slow-scale SSA, we separate reactions that are fast from those that are low, make the fast reaction reach equilibrium and then computes the slow ones.
- **Hybrid Simulation**, more or less the latter, combining ODEs with stochastic, so we update a part with ODEs and another with SSA in ***parallel***.






>[!info]
> 






---
# References

Lecture note for the exact approaches.

