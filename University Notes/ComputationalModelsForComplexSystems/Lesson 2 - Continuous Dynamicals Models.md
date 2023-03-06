Date: [[2023-03-03]]

Status: #notes

Tags: [[Complex Systems]]

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