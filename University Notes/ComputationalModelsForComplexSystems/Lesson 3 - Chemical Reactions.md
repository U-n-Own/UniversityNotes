Date: [[2023-03-10]]

Status: #notes

Tags: [[Complex Systems]], [[A.I. Master Degree @Unipi]]

# Chemical reactions methaphor

Group of molecules that changes state and conformation of the system they are in.
Different molecules, *reactants* and *products*. We will represent molecules as letters, $A,B,C, H_{2},O_2$.

Molecules float in a certain fluid the *solution*, typically syntax for chemical reaction is

$$
l_1S_{1}+ ...+l_pS_{p} \rightarrow^{k}l'_1P_1+...+l'_{\gamma}P_\gamma
$$

The constant $k$ associated with reaction is a positive real number, that will be our parameter, and describe the speed of the chemical reaction. 

## Types of reactions
![[Pasted image 20230310142703.png]]

For catalysis we have a $E$ an enzyme that help the reaction to happen. Chemical reaction appen at most between 2 molecules, if there are more we're only representing more reactions.

### Catalysis

The reaction happens in three steps : *binding*, that gives a molecule that combine enzyme with the other molecule, then this get trasformed (only the substrate), and then the enzyme is removed from the transformed molecule, and we have our final product.

There is a stochastic process modelling the molecules in a group, if these are involved in a reaction. When this does take place we can describe
![[Pasted image 20230310143408.png]]
The more reactants the more reactions. If we have different reactant each one of them contribute to the rate of the reaction.

Let's say that we have a reversible reaction and in this system we have 10 reaction so $10k$ in a direction and $20k_{-1}$

![[Pasted image 20230310143734.png]]

Over time for $2H_{2}+O_{2}$ into water we can see that the reaction that produce water is 'faster' than those that produces the single components. We cannot take two reaction and compare the kinetic constant $k$. Different constant in different reaction change!

- Measure of concentration is $\frac{mol}{L}$
- Measure unit of reaction rate is $\frac{mol}{L\times sec}$

How can we see more reactants we have, different measure of $k$ we will get.
![[Pasted image 20230310144416.png]]

For single reversible reaction we can compute easily the **Dynamical equilibrium** of that, what happens is that the rate of two direction equals each other and we can see that nothing will happen but in reality the reaction are happening at the same rate! To get the dynamical equilibrium for *single reversible* we can manipulate the double arrow to get the $\frac{k}{k_{-1}}$ rate
![[Pasted image 20230310144825.png]]
We can exploit some information to evaluate it, the quantity of each molecule, the sum of $A+C$ (by *Conservation properties*) will be constant over time! We know the initial values and we can solve it. 
eg
![[Pasted image 20230310145049.png]]

This is how our chemical systems evolves over time and reach the state of equilibrium. This was a very easy system with only one molecule produced. We want to traslate reaction into differential equation then apply solver to compute them.

![[Pasted image 20230310145332.png]]

## From reaction to ODEs

Now each molecule $S$ we construct an ODE where for reactions where $S$ is treated as reactant so it's ODE will have negative term, for the case of product will be positive.

![[Pasted image 20230310145816.png]]

## Reverse engineering ODE

We can infer the system from the reaction, can we do the other way around?

Here we have two molecules, and 4 reactions!
![[Pasted image 20230310152201.png]]

For example which reaction can represent $6X$? We're producing $X$, so it's positive and is a product and a reactant, so we can rewrite in this way
![[Pasted image 20230310152344.png]]

In some cases we could group the terms, the two terms aboes $XY$, we can write a simpler reaction that produces both the terms. This is only to simplify our reaction to get a simpler system of ODEs.

Focusing on individual event and giving them sense in what is happening is very suitable for modeling these types of system.


## From chemical reaction to Lotka-Volterra model

Chemical reaction are very similar to the model we saw describing population that prays and gets predated, reactants are our (preys), and predator are described by the molecules that can chemically react. 
This can be represented more generally in that way. When preys meets predator we have a `reaction` that is to consume a prey.
![[Pasted image 20230310153529.png]]


### Sir model reverse engeenered
![[Pasted image 20230310153847.png]]

This was our most complex model that describes the infections in populations over time.
Apply the rules we learnt from chemical reaction we get this.
![[Pasted image 20230310153949.png]]

What about the logistic equation?

### Logistic equation

We've seen logistic equation in [[Lesson 1 - Discrete Dynamical Systems#Can we predict equilibrium?]]
![[Pasted image 20230310154136.png]]
We can explain this model in reaction fashion
![[Pasted image 20230310154205.png]]

We can write it with two reaction one way and the other, forward and backward.
![[Pasted image 20230310154254.png]]

The steady state is actually $K$!

Next step will be to consider ***Stochastic simulation of chemical reaction*** taking into account probabilities, these are how in the real word chemical reaction works.3




>[!info]
> 






---
# References

Simulator for chemical reactions: to use a chemical tool, we have to derive from the chemical reaction into an ODE and let them solve, but there are tools that automatize this passage, for complex reaction will be very difficult model one. 
Some tools in [python](http://libroadrunner.org/)
