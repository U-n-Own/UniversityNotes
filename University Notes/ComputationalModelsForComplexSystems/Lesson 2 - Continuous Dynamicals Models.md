Date: [[2023-03-03]]

Status: #notes

Tags: [[Complex Systems]], [[A.I. Master Degree @Unipi]]

# Ordinary Differential Equations 

Since discrete update is an approximation and introduce errors, we want a more precise model
To models this we can use ODEs, these are applied as a continuous changes.

## Linear birth models: but continuous

Same problem of yesterday, density of population is described by $N(t)$, yesterday we considered a fixed timestep of update, we keep that assumption to simplify our model, at least at the start. For *discretization* when our $\Delta t$ tends to zero, the change becomes insignificant!

For that limit we get by rewriting 

![[Pasted image 20230303092905.png]]

But this is the derivative! So we know now that derivative of $N(t)$ is equal to that, where $r_{c}= \frac{\lambda}{\sigma}$, and the little c stands for continuous growth rate. 
What is important is that in this way we are describing that the ODE describes the difference between the population at the $\Delta t$ and the one that we got after that infinitesimal time, this has a very different meaning from before.

This is now our *Ordinary Differential Equation*:
$$
\dot{N}(t) = r_{c}N(t)
$$

Now we would want a **closed form definition** how do we do?
Before we can infer a certain solution and with induction we find a closed form, in these cases could be impractical or impossible, but this simple case is doable, let's see.

$$
\frac{\dot{N}(t)}{N(t)} = r_c 
$$

Since this term is derivative of $ln(N(t))$ and $r_c$ is the derivative of $r_{c}t+c$ for any constant $c$ we get

$$
lnN(t) = r_{c}t + c 
$$

This gives our closed form, that is in exponential form as we can see also in the plot
$$
N(t) = Ce^{r_ct}
$$

![[Pasted image 20230303094015.png]]

## Discrete and Continuous

Quantitatively they have the same growth: exponential, what changes is the role of growth rate![[Pasted image 20230303094157.png]]

## Radioactive decay description

Last time we modeled death, now let's see something more intresting where population can shrink. This is the idea of each molecule decays at a constant rate, so a mass decreases with a rate proportional to that mass itself. This can be described with this ODEs:

$$\dot{N}(t) = -d_cN(t)$$

A **negative derivative** means that what we measure is decreasing!
A closed form solution will be

$$N(t) = N(0)e^{-d_{c}t} $$

![[Pasted image 20230303094845.png]]

## Continuum logistic equation

So remember that $K$ corresponds to the population members that our environment can have, the *carry capacity* of the environment! Depending on the $K$ we can see what will happen to our derivative to understand if decreases or grows.

![[Pasted image 20230303094928.png]]

A closed solution analytical of this ODE is

![[Pasted image 20230303095140.png]]

If you see for the $t$ that goes to infinity the population stabilizes at $K.
The *equlibrium point* in the continuum case is different from what we've seen in discrete case

![[Pasted image 20230303095303.png]]

## System of ODEs

We can have like system of ODEs to describe more variables
![[Pasted image 20230303095627.png]]

For this system we don't show now a closed form we will see later how to get the graph
Now this results shows that males approaches zero and females start to approach the 100% of population.
Why this behaviour? Because now we're working with a derivative describing the systems, in discrete case we have a simple difference of $s_dM_t$ (the deaths of males).
Instead the male population has that factor with a *negative derivative*, here's why they decrease exponentially!
![[Pasted image 20230303095714.png]]

## Equilibrium point

In the continuous state is when derivative is zero, instead in discrete is when there is no change in population. We can see that $M(t) = 0$ 
![[Pasted image 20230303100116.png]]

### Numerical solutions

ODEs become very complicated sometimes, often ODEs are studied with **numerical solvers** and **simulators** usually they solve the *initial value problem* or Cauchy problem.
Now we see how to use iterative methods to solve ODEs.

![[Pasted image 20230303101557.png]]

Setting variables to initial values such that we can plot the variables over time and see how they behaviour

### Euler method 

Idea is to perform steps so that we approximate steps with the derivative, given a certain ODE

$$\dot{N}(t) = f(N(t))$$

This corresponds to approximate the solution with a recurrence relation

$$N_{k+1} = N_{k}+ \tau f(N_k)$$
![[Pasted image 20230303102155.png]]

As we can see we're approximating the solution with a certain error, what is the error?
The **local discretization error** is $|N(\tau)-N_1|$ and it's in the order of $O(\tau^2)$, like Taylor summation that is the in the case of First Order approximation, this error is quite big, also errors accumulates after $k$ steps at time $t = k\tau$ we have **Global discretization error** $|N(k\tau)-N_k|$ and is in the order of $O(k\tau^2)$ and since $k\tau = t$ is constant is only $O(\tau)$.

### More explicit methods.

A linear globar discretization error imply very small $\tau$ to be used, so computation becomes very slow.

Other methods have a global discretization error of high order eg $O(\tau^p)$ for some $p$ which is better when $\tau \to 0$.

Few examples:

- **Runge-Kutta methods** : $p = 2$ in original formulation but can be higher
- **Multistep methods (e.g. Adam methods):** extrapolate the value of next step from values of previous step

#### State of art methods

- **Self determine** the step size $\tau$ based on thresholds on local and global errors
- **Dynamically Adjust** the step size $\tau$ during execution (Adaptive Runge-Kutta)

## Instability and stiff systems

There can be problems if the process if fast for example for very high coefficients

$$\dot{N}(t) = -15N(t)$$

If step is too long the approximation that we do can go very far, if the equation is irregular we have to use small time steps.

These kind of systems are called **stiff systems** and there's no precision definition of *stifness*, intuitively since systems contains fast terms make us use very small term to slow them.
A solution is to use **implicits methods**

### Implicit methods

We perform a step when we use the derivative used at the end, since is very fast the derivative at the ends is closer to the one that is computed at beginning.
![[Pasted image 20230303103739.png]]
But how do we evaluate on the $N_{k+1}$ now we do not know it at the start, there is something in both size of equation we have to solve it at each step, this can be done but requires more work, we need some linear equation solver. More computation on this reduce the error, we use less iteration but slighly costly. But if the system is stiff this is quite compensated by that.
Methods discussed before exist in an implicit fashion, implementation of these require modelere to provide a **Jacobian Matrix** of the function, this matrix can be computed by the methods (Implicit Runge-Kutta...).

In this case our methds can self determine $\tau$ or dynamically adjust it. There are methods that can automatically switch from explicit and implicit methods and viceversa.

### Example of stiff system

[Code can be found in the slides](file:///home/vincentkun/Downloads/03-ContinuousDynamicalSystems.pdf)

![[Pasted image 20230303105432.png]]
![[Pasted image 20230303105456.png]]

![[Pasted image 20230303105512.png]]

## Famous ODEs models

In biology they appear usually, so it's better to know them. From now on we omit the $t$ in our equations so $X = X(t), \dot{X} = \dot{X}(t)$

### Lotka-Volterra equations 

Basically the prey-predator model and their interaction, we have mixel population of preys and predator and we analyze their behaviour over time, this has also been observed by *Lotka* as description of biochemical oscillator

Volterra observed that in Adriatic sea after WW1

- Ecologist observed an increase of population of some species of fish, and other species of fish were in lower number. The main consequence was the harvest of fish in the sea during the war. There was a link between these species, the **key idea** is if we have two population of preys and predator, if you reduce the number of predator, preys proliferate.

Describe population *density* with two variables $V$ preys, $P$ predators, 


$$
\dot{V} = rV \; exponential \; growth
$$
$$\dot{P} = -sP \; decay \; exp$$

So predators *hunts* prey and frequency of predation is related to the $P(find\;prey)$ so the rate of predation is proportional to both $P,V$. We use a coefficient $a$ that models the meeting results in an hunting (Probability to prey). Now we have to take into account derivative describing the frequency of something, rather than probabilities. But we can use similarly probability as frequencies as word in this context. Moreover once we consider the fact that predator can survive and reproduce, so denoting $b$ (rate of predation) the number of offsprings produced for each hunting, so more predator predate more they reproduce.

So now we can update our model with these rates.
![[Pasted image 20230309114636.png]]
Predator not only dies, but also reproduce, $aVP$ is called ***Hunting rate*** and $abVP$ ***Hunting-Reproduction rate***. We introduce non-linear terms that describe interaction between population.

### Playing with the model
![[Pasted image 20230309115056.png]]

Compute equilibrium is easy, we set the derivative to $0$. And we get two solution $V=0, P=0$ and $V = 1000, P=1000$.
What happens if we perturbate the number with different values? $V,P = 900$.

As we can see they start to grow togheter, than go over the equilibrium point and then preys start to decrease and predators follow them! 
![[Pasted image 20230309115437.png]]

Lotka observed chemical systems and observed oscillation, then he traslated the chemical reaction into differential equation that gave the same behaviour to this prey system.
These model has been used to describe markets (buyers and sellers), social system...

Let's use the solver for small values. We have a very fast increase in preys, followed by fast increase in predators. We get strong oscillations, if you add more details to the model complexity will grow and we lose track of which we want to consider. If with a simple model we can find a reasonable behaviour we can approximately say that these simplistic model work and have good properties. If we work with data we need a lot of data of all possible cases, in our case we want to formulate a function that descrine the system in the more simple way that approach it.
![[Pasted image 20230309115843.png]]


### Sir model

Used to model epidemic phenomena or their spread, we split the population in three parts

*Sir* stands for:

- ***Susceptible***: those who can get the disease.
- ***Infected***: those who have currently the disease, these can infect
- ***Recovered***: those who have recovered and cannot be infected anymore, also cannot infect

These model are often used to study spread of information in social networks. Our three variables $S,I,R$, describe ratios of each class of individual in the population.

**Assumptions**: Sum of $S,I,R$ is *normalized* to $1$. We don't have migrations, deaths, horizontal trasmission, one individual can infect 1 individual, (not graph like spread), Contacts are random (considering big population the interaction isn't random). People in one city cannot have contacts with other city, but we're doing assumption. After infection we can't infect anymore.

This is our initial model:
![[Pasted image 20230309122241.png]]

$\beta$ is the infection coefficient, everytime some $S$ being infected they became $I$ with this factor. $\gamma$ is the recovery coefficient, that describe tha rate of recovery of each infected, where $\frac{1}{\gamma}$ is the *time* that one indivisual have to recover. Basically is a frequency of recovery. $\beta SI$ is **Frequency of infection**, $\gamma I$ is **Frequency of recovery**, if the derivative is a constant we have a costant change. Multiplying by the value of the derivative. For example if $I$ increased at the previous step the growth of it becomes exponential!

Overve that $S$ can only decrease and $R$ only increase, but the dynamics of $I$ tells us if the disease is spreading or not, if $I$ increase there are more contagious than recovery so there is a great spread, instead if $I$ decreases we're seeing our population recovering basically. This is the case in *COVID-19* where we calculated the $r_0$ that essentially consist in $\frac{\beta}{\gamma}$.

Let's see how the model behaviours.
![[Pasted image 20230309123703.png]]

## Modeling influenza spread

Every winter we have a spread of influenza, we have some knowledge about flu, we're sick for a certain period of time of circa 8 days.
Our $\gamma = \frac{1}{8}$, or we can say that every week $12,5\%$ of population (infected) recover. A person with flu spread to $\frac{1}{5}$ of the persons that have contact with. And the epidemics last 120 days. 

Let's see with this numbers how the model solve the problem
![[Pasted image 20230309124244.png]]

$65\%$ gets disease with peak of infection at month 2 circa.

Usually these models are used to study how vaccines works and their effect. To do this we need to extend the model with some coefficients.

#### Adding birth and deaths

**Assumptions**: For sake of simplicity we want that population size be constant over time so birth and death in 10 years do not change so much, no vertical trasmission, newborns are susceptible.

$$N= S+I+R = 1 \; where \; \dot{N} =\mu -\mu N $$

We want to add a $\mu$ that is the growth rate of susceptible population and a negative term

Extended model
![[Pasted image 20230309124912.png]]

Now we have different cases for values of

1. $\frac{\beta}{\mu + \gamma} \lt 1$
2. $\frac{\beta}{\mu + \gamma} \gt 1$



**First**
![[Pasted image 20230309125047.png]]

**Second**
![[Pasted image 20230309125126.png]]


What happens when we vaccinate? 

We want further extend the model

#### Adding vaccinations

**Assumption**: We vaccinate newborns, not random people, we vaccinate a certain number of them. Let's call it $p$ these are in recovered state from the birth basically
![[Pasted image 20230309125531.png]]

Let's run some simulation with small number of $p = 0.1$
![[Pasted image 20230309125603.png]]

With $p = 0.5$
![[Pasted image 20230309125627.png]]

So if we vaccinate the half of newborn we can eradicate the disease!

We can compute analytically a certain threshold of vaccination to eradicate.

$$
p_{c} = 1 - \frac{\mu + \gamma}{\beta}
$$

Let's consider real disease like **Measles** (morbillo).
![[Pasted image 20230309125846.png]]

So for eradicate morbillo, we need to vaccinate $95\%$ newborns, so it's mandatory now to get vaccinated for example.
Spread of information follow a very similar model.

Next time we will see how stochastic models works, because some system in same initial state can change a lot because of stochastic nature of those systems.


```ad-todo
Consider what happens in the Lotka-Volterra model by adding vegetations.


Extend influenza model with vaccination

```


Lotka volterra phase portrait done myself: 100k steps eta
![[Pasted image 20230331175820.png]]

>[!info]
> 






---
# References

Chapter 2 of Guide to Numerical Modelling in Systems Biology
Mathematical and Computer Modeling of Nonlinear Biosystems

### Implementation of ODE solvers

- Python: scipy.interate.odeint and scipy.integrate.ode
- Octave: lsode
- C++: odeint
- Matlab: ode45, ode113...