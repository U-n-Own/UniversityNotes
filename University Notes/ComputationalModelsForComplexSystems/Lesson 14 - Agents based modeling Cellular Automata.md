Date: [[2023-05-04]]

Status: #notes

Tags: [[Dynamical and Complex Systems]],[[Cellular Automata]],[[Agents]],[[Environment]].

# Intro to agent modeling

We can have two or three dimensional enviroments, with inside agent following rules. These agent act on the environment itself. 
Similar to [[Lesson 18 - Intro to Reinforcement Learning]], but this is more biologically inspired, tissues in cells for example have an atonomous behaviour, or large animal colony like ants. These obeys to locally applied rules, simple but generate a global behaviour that .

![[Pasted image 20230505142745.png]]

## Modelling cellular systems

We can have grids, arrays or even tridimensional, for example we can even think to the **neighbourhood** behaviour

![[Pasted image 20230505142846.png]]


## State set and Transition rules

Thank to these we can model the behaviour of our system, how cell change state

![[Pasted image 20230505143047.png]]


## Boundary conditions

How far away the propagation of the rules apply

![[Pasted image 20230505143140.png]]

## Initial conditions

The initial state of a dynamical system matter, in fact a slightly change in initial state would be a great difference over time, even in small time step afterwards the system starts his "life"

![[Pasted image 20230505143246.png]]

### Modelling traffic

![[Pasted image 20230505143616.png]]

These are a specification of how cell evolves

![[Pasted image 20230505143832.png]]

We can see somewhat emergent behaviour from these simple rules being applied.
Can set some parameters to see what change to the parameter of initial condition? Yes, consider frequency of cars that arrives $\rho$ 

![[Pasted image 20230505144003.png]]

The idea behind considering *phase transition* is seen in [[Phase Transition]] systems like in Neuroscience (Brain Phase Transition) or even physics like [[Ising Model]].


## Formal definition of CAs

![[Pasted image 20230505144922.png]]

We can see that the possible configuration grows very fast with the type of changes we can do to these systems, so we introduce special rules (transitions).
![[Pasted image 20230505145007.png]]

## Wolfram's Rule Code

Wolfram studied a lot these cellular automata

![[Pasted image 20230505145243.png]]

An intresting discovering is that we can have 256 possible rules, and changing the rule we can change the behaviour, obtaining different emerging behaviour

We have pattern like Rule56 with a more organized schema, and Rule18 with unpredictable behaviour.
![[Pasted image 20230505145541.png]]

We can study different models, for example how people exit from a room, or how wildefires spread, and we can see how these evolves from simple rules: maybe winds blowing from a certain part favour fire in a certain direction and we can "simulate"

## Game of Life

This is quite famous cellular automaton, also Turing Complete, we can build computer running minecraft with this cellular automata. 
Here's rules are quite simple

- Birth: A white square become black (alive) if 3 neighboar are alive
- Survival: black remain black is there are two neighboar black
- Death: black becomes white if there are more than 2 neighboars


>[!info]
> 






---
# References

