Date: [[2023-05-04]]

Status: #notes

Tags: [[Dynamical and Complex Systems]], [[Discrete Event Simulation]],[[Simulation]].

# DES

We've seen **Stochastic Simulation** for example [[Lesson 4 - Randomness in Chemical Reaction#Gillespie's Stochastic Algorithm (SSA)]], stochastic simulation produces descriptions of possible behaviour of our systems, these approaches present some weaknesses, they are **not instantaneous** this is a strong assumption, in some cases we can have different distribution that represent our model, for example production of a certain number of proteins takes some time, fixed. Or we can event from gaussian distribution, with a certain average and variance. Some events can be related to some conditions that needs to be satisfied.

Determining when an event should happen require different way to perform simulation, Discrete Event Simulation can deal with all those issues.

We need to maintain an history of the process in an appropriate data structure, what do we use? Not only by maintaining state but also a list of distributions over time.

[example of customers]

We can consider different timing for each type of event: arrival of costumers, moving from queue to service or serving a costumer. Here **memoryless** property isn't assumed to be true. We've to do discrimination on costumers or updating the clock of the simulator.

## Description of DE simulation

[insert slide]

An **activity** is a sequence of **events**.
**Event notice** can be tought as a TODO list, coupling event with their time to happen.

### FEL 

[Slide]

Idea of simulation algorithm is to

[some java code]

>[!info]
> 






---
# References

