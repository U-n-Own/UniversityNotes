Date: [[2023-03-24]]

Status: #notes

Tags: [[Complex Systems]],[[A.I. Master Degree @Unipi]]

# Alternative Transition Systems

Today we will learn that certain TS are more complex, we can add probabilities to our sytems, the transitions will be modeled this way. These are things we encountered in [[Lesson 7 - Hidden Markov Model#Markov Chain and HMM]].


Let's start with an example:
![[Pasted image 20230324141943.png]]
Is this really $P(X=x_{i})= \frac{1}{6}$?
Show this
$$
\sum\limits_i^{\infty}\frac{1}{2}^{2i+1} = \frac{1}{6}
$$

## Extending TS with probabilities
![[Pasted image 20230324142600.png]]

Speaking of TS, we say that these are infinite, but markov chains are finite usually.

![[Pasted image 20230324142644.png]]
Probability to jump from a state to another is the following matrix, if we want to reach a probability .

A simple example
![[Pasted image 20230324142753.png]]

With as rows and columns all our state and possible transition, we will model also non transition with zero probability.

Initial state is modeled by a certain probability, we can assign a probability to a set of states to determine who could start, as we can see above. Our probabilities has to be one as sum, if one state has at least one outgoing transition those with only one have deadlocks.
![[Pasted image 20230324142947.png]]

### Trace in markov models
In this case traces are a bit more complex, each path has a probabilities, a path is defined by the multiplication of the probabilities that are in it.

![[Pasted image 20230324143229.png]]
### Reachability

A state is reachable in this case if there exist a path or a non-zero probability to reach it, we speak about all the paths.
![[Pasted image 20230324143535.png]]


How do we specify a probability to reach $s_2$ in our example?
![[Pasted image 20230324143945.png]]
Well speaking on this if we get stuck in the unique path that stays infinitely $s_0,s_1$ we are never getting to that last state and so we are doing $(0.99 \times 1)^{\infty}$

We can rewrite this as a *system of linear equation*.

![[Pasted image 20230324144233.png]]

This is quite simple taking solution if we assign 1 to the variables, so getting reached.

For the algorithm we saw before
![[Pasted image 20230324144421.png]]

This is our system of equations.


![[Pasted image 20230324144643.png]]

## Continous Time Markov Chains 

![[Pasted image 20230324144733.png]]

I can reason on complex behaviour by using reachability property more than one times. For stochastic simulation of chemical reaction we didn't used discrete distribution, we used the time as a continuous variable, now we will describe a stochastic rate and will see some *race conditions* between events. So we will have rate associated to transitions, for each pair of state we will have a positive real number describing frequencies of events happening, these is called transition rate matrix.


### Race conditions 
On average the first $s_1$ is slower than the one with smaller rate. These are to be considered as parameter of exponential distribution, the 2,3,4 are just the rates! So we can ask "how long we stay in a state", "what transition will be taken", this is really similar to Gillespies algorithm two steps. For $s_0$ we have an exit rate of 5. 
![[Pasted image 20230324145036.png]]

We can imagine using a simple prob. distribution, so $P(s_0,s_1) = \frac{2}{5}$. We can now describe a markov chain, called *Embedded Markov Chain*
![[Pasted image 20230324145419.png]]

### Embedded DTMC
![[Pasted image 20230324145550.png]]

After we normalize these and add self loop to state with no ougoing transition we get a description.
![[Pasted image 20230324145629.png]]

What we loose from going from continues time to embedded is the time, in the discrete we're not able to represent the time spent into a certain state. Remember we're intrested in *reachability* because knowing this we can construct some technique, so in DTMC we can define rechability by probability of reaching for example: "What is the probability to reach state $s_i$ in a time $t$?". If we know both these.

#### Reachability is intresting

We have a certain different in these two models of reachability

1. We can use Discrete markov if we are intrested to know the probability to reach a state indepedntly of time. 
2. We can use continuous 

### Uninfomisation

We choose a parater $q$ bigger than the sum of all rates.
![[Pasted image 20230324150243.png]]

What are we doing? We're dividing the time unit in steps of $q$ so imagine that we're dividing the rate by $q$ we're describing time transitions, obtaining discrete time markov chain where each step correspond to a unit of time, for example a path with 3 steps is 0.3 second of behaviour in the model. This is an approximation, the fact is that if we can transite rapidly maybe 0.001 second and one transition is happened but our steps are 0.1 each, so we make smaller time step to model better our  transitions.



Now we can reapply **probabilistic reachability** on the new markov chain, this is called ***Transient probabilistic reachability***
![[Pasted image 20230324150735.png]]


## Application to chemical reactions

Now we can work again on chemical reaction with this instrument.
![[Pasted image 20230324150837.png]]

We could now calculate the probability to reach one certain molecule quantity in a certain time after reaction is started.

> 



---
# References

Prism model checker tool presented.
