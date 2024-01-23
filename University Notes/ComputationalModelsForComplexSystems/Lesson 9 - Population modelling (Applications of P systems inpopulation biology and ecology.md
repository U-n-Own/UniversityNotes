Date: [[2023-04-13]]

Status: #notes

Tags: [[Dynamical and Complex Systems]]

# Modelling populations: P systems

P systems can provide simple, elegant and unambiguous notation for modelling populations, maximal parallelism is particularly useful for *reproduction stages* or *hibernation stages*. All individuals assume to have the same behaviour with some rules set and depending on the stage, for example in the reproduction we have rules fro reproduction instead in hibernation we have selection rules because weaker individuals die. Idea is to have rules set, usually we have one set, how do we get more stages? We will use the trick of have different production rule **rule promoters** to emulate this.

## Introducing probabilities

These are necessary because of the behaviour stochastic of population, so for example birth of male against females or choices these are relly important for small values of populations.

### A new variant of P systems : Mininmal probabilistic P systems (MPP)

MPP system is a tuple $< V, w_0, R>$ where 

- V is a possibiliy infinite alphabet of objects, with $V^*$ denoting the universe of all multisets having $V$ as support
- $w_0 \in V^*$ is a multiset describing initial state 
- $R$ is the set of rules

#### Algorithm for evolution step

Starting from a current state of the system, then we do the `maximal parallel step` application, the idea is to apply the rules sequentially, consuming products and when no rules can be applied we go to the next step.

The step that check if rule can be applied is the $u \subseteq x \wedge p \subseteq y$

![[Pasted image 20230413114000.png]]

At the end we can see that choosen rule obey to probability chosing among rules making them compete for the same object.

## Analysis techniques

We can do simulation or use statistical model checking

Here we have a population, a statistical model checker runs simulation of model use the results from simulation to construct DTMC that represent behaviour of the system, finally veryfing properties on the markov chain, we can do this in PRISM.

# Let's talk about frogs


The first ones are smaller, the ridibundus are bigger, these to can breed togheter dealing an `esculentus`

![[Pasted image 20230413114536.png]]
![[Pasted image 20230413114641.png]]


These frogs are diploid they get one copy from the mother and one from the father
![[Pasted image 20230413114736.png]]

In real lakes and ponds there are two different species coexisting, in wester europe we have a breed of the original and the hybrid togheter (Lessonae and esculentus). Is strange that these breed of individuals are common and they live togheter in a symbiotic way. This is yet not understood, why these two reaches stability? Maybe depends to initial conditions and can take years to have a graps on how these evolve.
![[Pasted image 20230413115010.png]]

## Hemiclonality

The esculentus have a strange behaviour of gametogenesis : **hemiclonal**.

- AI Generated
```ad-note

Hemiclonality: Hemiclonality is a phenomenon in which a population of organisms exhibits some degree of genetic similarity among its members, but also displays genetic diversity due to occasional sexual reproduction or mutation events. This means that the population is not completely clonal (i.e., consisting of genetically identical individuals), but also not completely sexually reproducing (i.e., with no genetic similarities among offspring). Hemiclonality can be observed in various organisms, such as plants, fungi, and animals. It is thought to providea balance between the advantages of both clonality and sexual reproduction.


One example of hemiclonality is found in some species of fungi, where asexual reproduction through spores is the primary mode of reproduction, but occasional sexual reproduction can also occur. This allows for genetic diversity and adaptation to changing environmental conditions while still maintaining the benefits of asexual reproduction, such as rapid population growth and the ability to colonize new areas quickly.
```

Hemiclonality: consist in the idea that we can keep an equal gametogenesis in which we get an exemplar that is a clone of the other.

![[Pasted image 20230413115146.png]]

## Reproduction table

Typically females $RR$ dies immediately, this is subject of this study.
What happens when we have the four type of frog? After a while we get an high number of exemplar in numerical advantage the $L_yR, LR$ and $LR$ 

![[Pasted image 20230413115535.png]]


If we leave hybrids alone they produce offsprings that are inviable they are stronger they should overtake other species but if they do, they will tend to disappear so what is the mechanism that make live them in symbiosis.

![[Pasted image 20230413115745.png]]


### Hypotesys 1: Sexual preferences
![[Pasted image 20230413120021.png]]
![[Pasted image 20230413115919.png]]

### Effect of preferences:

![[Pasted image 20230413120112.png]]

### Muller's ratchet

We can simulate how system evolves if the $RR$ is viable, hypotesis of death on that individual is that there is no combination, when we get an offspring with the same genes without recombination accumulating mutations can be saved by the other half of the cromosomes the $L$ ones, but if one exemplar has only the bad genes it dies, this phenomena is called **Muller's ratchet**. But is this necessary the death of those individuals?

## L-E complexes: MPP systems model

We've an objects for each chromosome, individual are of different types, for example $*$ for individuable with deleterious mutations and $°$ for mutation that are `safe`. This will allow us to explore the combination. The juveniles are the same, but we add a $^j$ superscript, for example $R_*,R_*^J$ will die more likely, then we have $V_{ctrl}$ that is the stage chosen the stage change the possible set of rules.
![[Pasted image 20230413122824.png]]


### Evolution rules

Using maximal parallelism: given a male $x$ and female $y$ and juvenile $y$ of any type we have a rule that produce it, this schema is followed only if the object $REPR$ is present.

For the rules we multiply the rates of female and males like in chemical reactions, then we use a constant to model sexual preferences $k_{mate}(x,y)$, and another constant $\frac{1}{k_{o\_kind}(x,y)}$ this tells us how many types of different offprings from the genetic point of view we have.

![[Pasted image 20230413123217.png]]

Here we can see that adults can die, juvenile can become adults.

Where our function $g_x(w)$ is a logistic function that integrates $cc$ (carrying capacity), so a rate for the survival that depends on strenght of individual and the rate of death is $g'_x(w) = 1 - g_x(w)$. 
![[Pasted image 20230413123549.png]]

Then we have rules for stage alternarting, we have three reproduction stages followed by 1 selection stages, maximal parallelism ensures that these are always applied.
![[Pasted image 20230413123827.png]]

### Global description

Thanks so maximal parallelism and our defined language we can express in a concise way complex models. We've got 9 rules and 2 stages.
![[Pasted image 20230413124019.png]]


## Simulations

![[Pasted image 20230413124118.png]]

### Statistical model checking : probability of extinction

![[Pasted image 20230413124334.png]]


## More intresting model checking : ridibundus viable simulation.

So let's say that $RR$ do not die instantly and can reproduce like the others.
![[Pasted image 20230413124505.png]]

What we get is this

![[Pasted image 20230413124736.png]]

![[Pasted image 20230413124824.png]]

![[Pasted image 20230413124855.png]]

After 35 years ridibundus females overtake other population but then all gets extingued. 

How can we explain this? Since they are all female when they overcome other populations they remains without any exemplar of male to reproduct.

## More complex scenario.
![[Pasted image 20230413125211.png]]

We introduce a small amount of ridibundus males and females.
We see that they overcome the other make them disappear and become the dominant species.
![[Pasted image 20230413125245.png]]

### Extinction

Again why this happens? Since hybrids produce offspring of type ridibundus, and as combination in the table they are more probable to born as offspring so they dominate other species.
![[Pasted image 20230413125332.png]]

![[Pasted image 20230413125458.png]]
>[!info]
> 

# Conclusions

We use P systems as notation for population models and simulation+model checking for analysis, the case 




---
# References

[Paper and slides presented here](https://arpi.unipi.it/handle/11568/851713)
