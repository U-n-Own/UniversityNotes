Date: [[2023-03-23]]

Status: #notes

Tags: [[Dynamical and Complex Systems]],[[A.I. Master Degree @Unipi]]

# Simulations

We could perform many simulation without getting the same rare event for example 1 over $10^6$ times probability to happen.

## Definition of Transition System

![[Pasted image 20230323112619.png]]

today we will see how we can see a transition system in a composition without adding too much complex stuff, so without probability or random event that make us transite to a state to another. *Markov chains* basically.

What we're constructing is a graph, possibly infinite graph because we have infinite states. Transition can be non-deterministic, so from one state we can go in more than one state. Another example are Non-Deterministic Automata in Computer science. Non-deterministic is different from stochastic because there's no criteria to pass from state $A$ to $B \lor C$.

### Trace

![[Pasted image 20230323113311.png]]

A trace is one possible match or `path` the system can take, for example in a soccer match we go from state $(0,0)\mapsto(1,0)\mapsto(1,0)_{end}$

A *Trace* is maximal if $t$ is infinite or if the state $s_n$ has no changes from the last transition
![[Pasted image 20230323113613.png]]

### Reachability

![[Pasted image 20230323113658.png]]

How do we check if a state $s$ is reachable? We can use Breadth-First-Search on the fly generation of state (lazy), if the graph is infinite for example.

### Kripke structures

Very common transition system, set of boolean variables, or a set of boolean `propositions`.
![[Pasted image 20230323113853.png]]

![[Pasted image 20230323114002.png]]

By describing the states using only boolean we can use integer to describe the states, so for example we can use bits to describe these state, this allow to have compact representation. Once we have it we can say that the state that says that error is true is rechable, so we want that maybe is not reachable certain state.

### Over set of variables

![[Pasted image 20230323114418.png]]

If we want to have a richer set of variables of different type we can represent it this way, all the possible combination we can give to our variables is a state.
![[Pasted image 20230323114521.png]]
If these are continuous set like Real numbers, this becomes very hard, we restrict this to a smaller infinite such that of the integers.
For example if we have a transition system with continuous variables between each of them there is an infinitely possibility of state, and these states are much denser. For the soccer match instead the score is a integer number.

#### Queues

![[Pasted image 20230323114858.png]]

We can se that initial state has empty state, and after if becomes `busy` we increase the value of $q$ by $1$, until we reach the limit of our queue.

### Specify a transition system

Specify transition system means to define their behaviour so a set of rules, transition rules. $if-then$ rules.

$$guard \mapsto update$$

where 

- Guard is a conjunction of condition on the state variables for instance pick a state that is our set we conjunct all the state that satisfies these conditions, our conjuction.
- Update is a conjunction of assignments to these state variables.

So with a single rule we can define infinite transitions.

Our example of soccer game, we can see that we have infinite transition toward the top of the diagram adding a score to the first team for example. So this specify an infinite transition system in a compact way. 
![[Pasted image 20230323115418.png]]


Let's see how this works for queues

We can specify this rules
![[Pasted image 20230323115631.png]]

Note that second transition doesn't change the busy to true it only checks it, busy becomes false only if busy is true and $q=0$.


## Labeled system




## LTS


![[Pasted image 20230323120744.png]]

## Why transition labels
![[Pasted image 20230323120823.png]]
These are really good for model parallel systems

![[Pasted image 20230323120959.png]]

## Covfefe machine 

Let's pick an example of a coffee machine taking two coins before giving covfefe to the user
![[Pasted image 20230323121041.png]]

How do we couple the two components? This would be description of our system behaviour. Well what we can do a description of the whole system, we could take all the possible combination of these values.
![[Pasted image 20230323121147.png]]


The possible combination diagram gives us this $4\times4$ states.
![[Pasted image 20230323121429.png]]

We can add further transition that are basically the same we have but take different traces
![[Pasted image 20230323121735.png]]

But now we can remove some transition and restrict some of the transition, now we consider to remove all non-$\tau$ transition because aren't reachable from initial state. Also we remove $\tau$ transition unreachable from initial state, consider now different behaviour like user with only 1 coin.
![[Pasted image 20230323122028.png]]
This is what happen!

### Adding another user

What happens if we add two users? They will compete to get the covfefe

After merging and removing unreachable state we get this

![[Pasted image 20230323122233.png]]

The first user perform his steps and in the end both users will reach the state drink, one will drink before the other. For example the central unit we have the two user at previous step both insert a coin and the machine is ready to prepare coffee, but they are stuck because want to insert the second coin (both).


## Lesson learnt

State space is huge usually, we will see techniques that can deal with big transition systems.

>[!info]
> 






---
# References

