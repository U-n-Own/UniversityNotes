Date: [[2023-03-02]]

Status: #notes

Tags: [[Dynamical and Complex Systems]], [[A.I. Master Degree @Unipi]]

# Modeling systems

The intro lesson was about how we can model system, so using math with recurrence equation or computational models like **Petri Nets** or Cellular Automata.
The *dynamics* or behaviour of system will be described by some *transition system* so we pass to one state to another given an event that happened, and some of those transition are deterministic, so we can model transition with probability distribution sometimes. This is the semantic of a modeling language
![[Pasted image 20230302112638.png]]
These can be seen as well as *Markov Chains*, once we define these representations, we can do simulation, model checking.

# Discrete Dynamical Systems description

Discrete steps in which the values descripting the systems updateds, the mathematical approach is to use differential equation. This is an approach used to describe the computational cost of a function that is recursive.
The variables of interest are number of individual in population, at the start we will use a single variable to characterize this, so talking about *omogeneous population*.
When we will describe interaction between individuals we can see that chaotic behaviours emerge.

## Linear birth model

Let use $N(t)$ the density of a population at time $t$, we use density instead of size because can be generalized per individuals per unit of space. Independently of size of environment.
We want to predict the future density $t + \Delta T$ that is $N(t+\Delta t)$.

We will do some assumptions of enough food and space is supplied no death in the interval we want to analyze and children do not reproduce in the interval we are taking at that step.
But we know that each individual has $\lambda$ children every $\sigma$ time units.

![[Pasted image 20230302114145.png]]
Let's take an example like above.

- Bacteria grows and duplicate, so we have to use a $\Delta t$ that is smaller or equal to 20 minutes to see the duplication has occurred.

Let's define the recurrence relation as

$$
N(t + \Delta t) = N(t) + \lambda \frac{\Delta t}{\sigma} N(t)
$$

where $\frac{\Delta t}{\sigma}$ is the number of children that will be born in the interval $\Delta t$ and is called the *birth moments*.

We can rewrite in this way
$$
N(t + \Delta t) = \color{red}(1 + \lambda \frac{\Delta t}{\sigma}) \color{blue}N(t)
$$

The red part is a costant
![[Pasted image 20230302114835.png]]

From that equation we described we can derive a **discrete model** in this way, since the time step is fixed, using notation of sequence theory: $N_t$ = $N(t)$

$$N_{t+1} = r_dN_t$$

$r_d$ is the birth rate
![[Pasted image 20230302115144.png]]
Now we can se very clearly
![[Pasted image 20230302115221.png]]

For female fish population with a time step of 1 year the birth rate become 

$$N_{t+1} = 25N_t$$
![[Pasted image 20230302115345.png]]

So we can run simulation. But knowing it we can calculate the general term of system in a non-recursive way so to define non recursivily $N_t$ 

We can write 
$$N_1 = r_dN_{0}$$
$$N_{2}= r_dN_{1} = r_dN_0^2$$

And we can do an induction proof:
![[Pasted image 20230302115649.png]]

So the shape of closed solution we got suggest that $N_t$ is esponential in $t$ and it's describing *exponential growth*.

## Phase portrait

This is another tool we can use that is a graphical representation of the values in the time step by step, you draw the bisector of the diagram and in red the recurrence relation as a function.

$N_{t+1} = 2N_t$

![[Pasted image 20230302120023.png]]

Starting in the bisector we can bounce on the curve of the recurrence relation, and this is sorta a periodic behaviour.

## Adding deaths

Adding ingredient to make model more complex, now we have a certain number of population that dies.


$$N_{t+1} = r_dN_{t}- s_{d}N_t$$

where $0 \leq s_{d}\leq 1$

We can rewrite recurrence relation with the term ($r_{d}-s_d$), calling this term *net growth rate*, now we obtain a new recurrence relation, let's call it $\alpha_d$

![[Pasted image 20230302121521.png]]
![[Pasted image 20230302121539.png]]
![[Pasted image 20230302121554.png]]

## Adding Migration

Another ingredient is migration to take into account, *incoming migration* $\beta$ is a costant that models that some individuals, usually positive.

**General term**:
$$
N_t = \alpha_dN_{t-1} + \sum_{i=1}^{t-1} \alpha_d^i\beta
$$

We have to model that migrants have children that starts to grow with the others.

![[Pasted image 20230302121857.png]]

Linear growth for $\alpha = 1$ due to the linearity of migrants.
![[Pasted image 20230302122040.png]]

For smaller values of $\alpha$, the population in these case reach a *dynamic equilibrium*
![[Pasted image 20230302122139.png]]
This depends on the value of $\alpha, \beta$ and general growth the red line has both the events happening but the variables reached an equilibrium, that is independent on the $N_0$.

## Can we predict equilibrium?

Analitically we replace $N_{t+1}$ with $N_t$ in the recurrence equation.

$$N_{t}= \frac{\beta}{1-\alpha_d}$$

That will be our equilibrium point, of course in the case of $\alpha = 1$ the formula doesn't hold, we have constant growth in that case without considering $\beta$

## Interaction and non-linear models

We've seen linear models until now, let's consider non-linear in which we have interaction between population, and lets introduce ***Logistic Equation***, let's assume resource is modeled for interaction. To introduce $K$ that is the carrying capacity

$$
N_{t+1} = r_dN_t(1-\frac{N_t}{K})
$$

The more population is big more the role of $K$ will become substantial.
![[Pasted image 20230302123711.png]]

The population reaches an equlilibrium that depends on $K$, reaching this *dynamic equlibrium* we can also call it *saturation* if we considere these as molecules in a room.

Now let's play with the *growth rate* parameter an let's see what happens
![[Pasted image 20230302124019.png]]

At the right we have a *Phase Portrait* the red curve is the recurrence relation.
![[Pasted image 20230302124202.png]]
Now with higher $r_d$ we got a lot of oscillations but every time at the same values, this point is an **attractor**, but increasing more we get more sustained oscillation with different period.
![[Pasted image 20230302124323.png]]

Let's increase even more!
![[Pasted image 20230302124356.png]]

Strange behaviour occours and with $r_{d}=4$, we get chaotic behaviour
![[Pasted image 20230302124501.png]]
The increase of number of attractors is described by a geometric relation, the distance between bifurcation decreases geometrically: $\frac{dist_{i}}{dist_{i+1}} \approx 4.7$ this is called *Feigenbaum constant*.

## Systems of recurrence relations

We want to describe more complicated things in the environment and in the population, but we have males and females. Females compete with Males for resources and suppose that males die because they fight. We get something like this
![[Pasted image 20230302125115.png]]
Let's run a simulation and we can clearly see that males approach a number that is smaller than females because of the term they have that is modeled for their deaths
![[Pasted image 20230302125159.png]]

# Lesson learnt
We have to introduce approximation, such that during timesteps nothing happens, this for Discrete at least, in fact Discrete dynamical models are limited in some sense, in order to reduce approximation we must smaller steps 

$$
\lim_{\Delta t} -> 0
$$


## Exercise 

Split population into adults and children, and consider children become adults in 3 years, so we have to model the rate $\gamma$ of trasformation of children into adults, suppose also that children don't die. 


```jupyter

# Given a population where beta is how many children per adults every offspring, 
# and children don't dies, children became adults after 3 years.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

N = 1 # initial population
t = 100 # time units: 1 time unit is 1 year


beta = 2
deaths = 0.9

K = 25 # Ratio of occupancy
# Suggestion use more variables for children at 1 step, second step, 
# Lets give children state, 1: Children, 2: Adolescence, 3:Adults, 

# time to become adults
# every children grow to the next step every year.

def population(N, t, beta, deaths, K, children_s0, children_s1, children_s2):

    Nt = np.zeros(t)
    Ntot = np.zeros(t)
    Nt[0] = N
    Ntot[0] = 0
    
    CarryingCapacity = True
    
    # Use different variable for children and adults
    # Use a systems of equations to solve the problem
	# Rule: after an epoch t, children_s1 becomes children_s2, children_s2 become adults
    # Adults generate children of current epoch as children_s0, 
    # prev step children_s1 becomes s2, prev step s2 becomes adults
    
    # Loop for adults
    for i in range(1,t):
        
        # Population growth: adults + children
        
        children_s2 = children_s1
        
        children_s1 = children_s0
        
        # Modelling carrying capacity in the environment we add K
        # K is the maximum population the environment can support
        # The birth rate is modulated by Nt/K
        if CarryingCapacity:   
            children_s0 =  beta*Nt[i-1]*(1-Nt[i-1]/K)
        else:     
            children_s0 =  beta*Nt[i-1]
        
        # Population growth: adults
        Nt[i] = Nt[i-1]+children_s2 - (deaths*Nt[i-1])
   
        Ntot[i] = (Nt[i-1] + children_s0 + children_s1 + children_s2)


        if False:
            print("Total population at step", i, "are", Ntot[i])
            print("\n")
         
    return Ntot

def plot_populations(Nt):
    """ 
    Plot different populations
    Nt is a lsit of populations to plot
    """
    plt.figure(figsize=(10,6))
    for i in range(len(Nt)):
        plt.plot(Nt[i])

    plt.legend(['Nt', 'Nt2', 'Nt3', 'Nt4', 'Nt5', 'Nt6'])
    
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()

def plot_populations_seaborne(Nt):
	"""
	Plot different populations
	Nt is a lsit of populations to plot
	"""

	sns.set_theme(style="darkgrid")
	plt.figure(figsize=(10,6))
	# Change resolution of the plot
	sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})
	sns.lineplot(data=Nt, palette="bright", linewidth=2.5, dashes=False)
	plt.xlabel('Time')
	plt.ylabel('Population')
	plt.show()

Nt = population(N, t, beta, deaths, K, 0, 0, 0)
Nt2 = population(2, t, beta, 0.9, 40, 0 ,0, 0)
Nt3 = population(2, t, 1, 0.2, 40, 0, 0, 0)
Nt4 = population(2, t, 1.5, 0.9, 40, 0, 0, 0)
Nt5 = population(3, t, 1, 1, 50, 0, 0, 0)
Nt6 = population(25, t, 0.2, 0.3, 50, 0, 0, 0)


plot_populations_seaborne([Nt, Nt2, Nt3, Nt4, Nt5, Nt6])

```




>[!info]
> 






---
# References

