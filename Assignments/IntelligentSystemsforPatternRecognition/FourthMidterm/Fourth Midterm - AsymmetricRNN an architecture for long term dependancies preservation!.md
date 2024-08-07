
#### Author : Vincenzo Gargano

## Introduction

In this paper review i'm going to analyze a bit of theory behind the models proposed and the rather intresting connection between  *Dynamical Systems* and *Recurrent Neural Networks*. Then we will see how they developed this architecture (heavily studied in other ways by other scientists that tried to solve the problem) and then what is the solution proposed in this paper. 

Before going into this paper my objective is to do a little introduction of the things we will meet a long the way, to give the readers some basis to understand all the concepts that comes to play, as a Computer Scientist i was not very familiar with some of the things that were treated here, so i was really enthusiastic on learning more about this topic.

```ad-quote
Explaining mathemathics is like a journey, we have to reach a destination, but we cannot keep going without looking at the landscape, that's why important deepen knowledge about new unseen things.
```

## Dynamical Systems review

As Steven Strogatz says in his book:  

```ad-cite
*"Non linear systems are really difficult to solve analytically, linear systems are easy because they are decomposable, the idea is like those of signals that can be decomposed using Fourier Analysis, instead non linear systems are everywhere in nature, and hearing two beautiful songs don't get us twice the fun."*
```


[[Dynamical and Complex Systems]], are described by a mathematical model called [[State-Space Model]], there we think in terms of set of *state variables* whose values (at any instant of time) can be used to predict future evolution of this system.

I did a little discussion on why RNNs are dynamical systems in the previous Assignment but the juice is that these are systems that work on sequences of any kind and time is sequential...

Let $x_1(t), x_2(t)...x_N(t)$ be the state variables of a `nonlinear dynamic system`. Non linear dynamical system are usually difficult to analyze due to the presence of oscillations, stable cycles or chaotic behavior. That do not emerge in linear dynamical systems. 

So the dynamics of large class of nonlinear dynamic systems may be written in this form

$$
\frac{d}{d_t}x_j(t) = F_j(x_j(t)) \;\;\; j=1,2,...,N
$$
Where function $F_j(\cdot)$ is a non linear function of it's arguments

We could rewrite this in a vector compact form

$$
\frac{d}{d_t}\mathbf{x}(t) = \mathbf{F}(\mathbf{x}(t))$$

Where now non linear function $\mathbf{F}$ is vector valued, each elements of which operates on the state vector $\mathbf{x(t)}$. For the remaining part of this review i won't use bold character since all we will discussed will be implicitly (to lighthen notation) in vector form because of dimension of our RNNs.

### On convergence of dynamical systems

What we usually do in the context of non-linear system, in order to study some of properties of our intrest, is to do linear approximation on a certain `point of equilibria`.
So by converting our RNNs into ODEs and then analyzing the behaviour around that point we can gain a lot of insights about what the system's properties are, whether is unstable or stable and so on.
But what are we intrested in? The answer is the *Jacobian Matrix* and it's *eigenvalues*.

Next i'm going to introduce some other concepts that are used in the paper, so let's dive further in.

### Divergence theorem

Intuition is the key for everything and i think that visualization of our systems is what makes them really easy to understand.
the divergence represents how much a vector field "*spreads out*" or "*diverges*" from a point. The divergence theorem relates this divergence behaviour to the net flow of the vector field across the enclosing surface.

```ad-important
If the divergence $\nabla \cdot F(x)$ (which is a scalar) is zero, the system is conservative, and if it negative the system is dissipative.
```

For example what happens if a RNNs is dissipative?
In the worst case as also mentioned in the paper we could lose previously stored information in the forward step.

### Attractors

Thinking to dissipation we can picture it as a lose of energy over time of our system, in this case *Attractors and stable points*.

[[Dissipative systems]] are characterized by presence of attracting sets or manifolds of dimensionality lower than the state space. For a [[Manifold]] or **attractor** we can have stable trajectories, these represent the *equilibrium states* of a dynamical system.
Of course an attractor doesn't imply a steady state we could have cycles repeating continuously and as we will see this is 
what we're intrested in the case of learning long term dependancies.

### Hyperbolic attractors

Consider a point attractor whose nonlinear dynamic equations are linearized around the equilibrium state, let $A$ be the Jacobian evaluated, the attractor is said to be hyperbolic attractor if the eigenvalues of the jacobian have all absolute values less than $1$. Hyperbolic attractor are particularly intresting for [[Lesson 14 - RNN Sequential Models and Gated RNN#Gradients problem discovered]] : **Vanishing Gradient**.

```ad-note
Since gradients tells us the *direction* and *magnitude* of the changes in the system state variables.
```

Since the recurrence matrix $W$ of the Hidden-to-Hidden part stores the memory of our network so what information are stored and what discarded : this is true for gated architectures but not for the Vanilla because of the Constant Error Carousel effect we've seen.

### Stability of equilibrium states

Considering now an autonomous dynamic system described by state-space equation we've seen. A constant vector $\mathbf{\overline{x}} \in \mathcal{M}$ is said to be an **Equilibrium point** or **Stationary state**. 

If this condition is maintained.

$$\mathbf{F(\overline{x}) = 0}$$

Where $0$ is the null vector. Equilibrium state is unique? Not everytime.

So when $\overline{\textbf{x}} = x(t)$

In order to understand the **Equilibrium condition** suppose that the non linear function $F$ is smooth enough (**Lipschitz**) for state-space equation to be linearized in the neighborhood of equilibrium state.

$$
{x(t)} = \overline{x}+\Delta x(t)
$$

If we make a small deviation in $x(t)$ from $\overline{x}$, we may approximate it in this way.

$$
F(x)\approx\overline{x}+A\Delta x(t)
$$

Where $A$ is the **Jacobian** of nonlinear function $F$ evaluated at $x = \overline{x}$

we could define an equilibrium point as follows:

$$
\frac{d}{dt}\Delta x(t) \approx A \Delta x(t)
$$

Only if the Jacobian is invertible, this is sufficient to determine the local behaviour of the trajectories of the system. 
If Jacobian isn't invertible the nature of equilibria depends on eigevalues of the real part and they must be negative.

### Defining Stability

This section is maybe the most important in order to understand the problem and the solution proposed in the paper.
Let's get better definition of `stability`, `convergence` of equilibrium states.

1. Equilibrium is *uniformly stable*, if for any $\epsilon$ there exists another constant $\delta = \delta(\epsilon)$.

$$
||x(0) - \overline{x}|| \lt \delta
$$

$$
||x(t) - \overline{x}|| \lt \epsilon
$$

2. The equilibrium state is said to be *convergent* if there exist positive constant $\delta$ such that condition

$$
||x(0) - \overline{x}|| \lt \delta
$$

implies that

$x(t) \mapsto \overline{x}$ as $t \mapsto \infty$ 

Meaning that if the initial sate $x(0)$ is close to equilibrium state $\overline{x}$, then the trajectory describe by state vector $x(t)$ will approach the equilibrium state.

3. The equilibrium is *asymptotically stable* : if both stable and convergent.
4. The equilibrium is said to be *globally asymptotically stable*: if all trajectories converge to equilibrium. 

Where for *trajectory* we mean the direction "drawn" by our system trough time, it's a function that maps a time dependant variable to the state assumed at a certain time, for each time $t$. We will see these trajectories later in the paper 

Last definition let us know that as the time passes and reach infinity we will get to a single steady state for any initial configuration.

## Lyapunov's Theorems

Now the objective is determine stability, we could try all possible state space equation but the apporach is not doable, an elegant approach is founded by Lyapunov. We can leverage on a function called `Lyapunov Function`. These theorems describe autonomous non linear dynamical systems.

1. Theorem: The equilibrium state is **stable** if in a small neighborhood of $\overline{x}$, there exists a positive-definite function $V(x)$ such that its derivative with respect to time is negative semidefinite.
2. The equilibrium state is **asymptotically stable** if in small neighborhood there exists a positive definite function $V(x)$ such that the derivative is negative definite in that region.

The function $V(x)$ that satisfies the requirements of these teorems is called Lyapunov function for that equilibrium state.
Of course what we are intrested is to have *stabiity* rather than *asymptotically stability*. See [[#Simulating RNNs dynamics]] table below.

## AntisymmetricRNN 

The paper present this new architecture ***AntisymmetricRNN*** that is able to capture long-term dependening using stability property of its differential equation, as we discussed previously. 
Previous methods used were to use for initialization the following matrixes: $I$ or $Q$ along the introduction of Gating and layer normalization.

#### Definition 1 - Stability in RNN

They define like this: a `solution` $h(t)$ with initial condition $h(0)$ as *stable* if for any $\epsilon > 0 \; \exists \; \delta > 0$ such that any other solution $\tilde{h}(t)$ of the ODE with initial condition $\tilde{h}(0)$ satisfy the fact to be **Uniformly stable** as we seen in the section [[Fourth Midterm - AsymmetricRNN an architecture for long term dependancies preservation!#Better defining Stability]].

We need uniformly stable network : that if a small perturbation let's say $\delta$ on the initial state $h(0)$, the effect of this perturbation on the states that follows won't be bigger than a certain $\epsilon$. The eigenvalues $\lambda$ of the Jacobian matrix will play a central role in stability, as we know from the Lyapunov Theorems. (i hope need to double check).

We have a stable solution of the ODE if:

$$
max_{i=1,2,..n} \; Re(\lambda_i(J(t)))\le 0, \;\; \forall t \ge 0
$$
We define the $i-th$ eigenvalue because if that the real value of eigenvalues is smaller than 0, we get a `Lossy system` that is not a thing; we want ideally be really near the zero for each eigenvalues.

This condition is called *critical criterion*, here the system preserve long-term dependancies of the inputs while being stable, this is the pivot, longer we maintain this property the more dependancies we can "store" in our RNN recurrent weights.

$$
max_{i=1,2,..n} \; Re(\lambda_i(J(t))) \approx 0, \;\; \forall t \ge 0
$$

### Stability and Trainability

Trainability of our nets are important, how do we connect it to the stability of it's ODE?

Since we're studying the `sensitivity` of a solution, how the solution changes with changes in initial conditions.

How we can see in the equation above by differentiating the ODE of our RNN

$$
\frac{d}{dt} \left( \frac{\partial h(t)}{\partial h(0)} \right) = J(t)\frac{\partial h(t)}{\partial h(0)}
$$

We're characterizing what happens trought time to the changes of the current state given changes in initial condition. This is equal to our jacobian that represents the sensitivity on the current state $h(t)$, with respect to the previous one, by multipling $J$ with that change this allows us to se how the initial state will have effect and propagate through the RNN.

We'll call $A(t): \frac{\partial h(t)}{\partial h(0)}$ the $J$ of an hidden state $h_t$ with respect to the initial hidden state $h_0$. When the *critical criterion* is met, so $Re(\Lambda(J) \approx 0)$. And thus we get that that the magnitued of the $A(t)$ is constant in time, this means that we have no exploding nor vanishing gradients. This is why we want a system remains *Marginally Stable*! Our trajectories may oscillate or remain stationary. We want a **Neutral stability** to avoid the two scenarios we can also say to have a center manifold.

```ad-caution
When we consider a linearized part of our system we could not conclude that non linear effects can still be a problem.

```

### Antisymmetry is all you need

The thing is we want to design ODEs that satisfy that criterion. So we introduce the **Antisymmetric Matrix**: a square matrix whose transpose equals it's negative so

$$
M^T = -M
$$

Antisymmetrical matrices are a building block for a stable recurrent architecture. Why?

The property of interest is that all eigenvalues of $M$ are imaginary. So we have no real part! They are zero.

$$
J(t) = diag[tanh'((W_h-W_h^T)h(t)+V_hx(t)+b)]
(W_h-W_h^T)$$

The entries of this matrix are the derivatives of the activation, bounded in $[0,1]$, so jacobian changes smoothly overtime. 

Finally the paper presents a discretization of that ODE, and call this : **AntisymmetricRNN**

$$
h_t = h_{t-1} + \epsilon tanh((W_h-W_h^T)h_{t-1} + V_hx_t+b_h)
$$

Our $\epsilon$ would represent the small step we do when approximating. Since antisymmetric matrix has less degrees of freedom we can use the triangular matrix, and makes parameter efficient this net.

### Diffusion regularize stability

This along the introduction of Antisymmetric matrix is the at heart of this architecture and of course is an hyperparameter.
Now we have a stable ODE, but discretization can be still unstable, so we need a stability condition on the discretization method. 

$$
max_{i=1,2,...,n} |1 + \epsilon \lambda_i(J_t)| \le 1
$$

The ODE defined previously is incompatible with the stability condition of Euler method because the eigenvalues are all imaginary that quantity is greater than 1 and this make our AsymmetricRNN unstable.

One way to fix this is adding *Diffusion* to our system, we subtract a small $\gamma > 0$ from the diagonal elements of the transition matrix, so we need to add this hyperparameter to ensure stability. 

Maybe this is not true rewrite it

```ad-important
This diffusion hyperparameter can be seen as a "regularizer" (in a broadher way) for the vector field but it's not really preventing overfitting, that my intuition: by applying diffusion we **constrain** the vector field to be stable.
```


### Gating Mechanism

Gating is used a lot in RNNs, each gate is a single layered network taking a previous hidden state and data at that time instant $x_t$, then applying a sigmoid activation for example LSTM cell use three gates, as we saw in lesson [[Lesson 10 - LSTM]] and [[Lesson 14 - RNN Sequential Models and Gated RNN]]. 
Some ablation study noticed that some gates are crucial for the performance of the LSTM, so how can we incorporate LSTM into AntisymmetricRNN, such that the critical criterion holds?

Well they propose adding an input gate $z_t$ to control the flow of information into the hidden states

Defined

$$
z_t = \sigma((W_h-W_h^T-\gamma I)h_{t-1}+V_zx_t+b_z)
$$

and $h_t$

$$
h_t = h_{t-1} + \epsilon z_t \odot tanh((W_h-W_h^T -\gamma I)h_{t-1}+ V_hx_t+b_h)
$$

So the $z_t$ uses Sigmoid as activation and the hidden state used Tanh. Effectively behaving as we seeen at lesson the sigmoid returning some values from $[0,1]$ and we use Hadamard product to apply the gating. 

This is how is defined the Input gate in usual LSTM architectures (and also update gate in GRUs)

$$I_t = \sigma(W_{Ih}h_{t-1}+W_{Iin}x_t+b_I)$$

So we get a new shared weights matrix that is antisymmetric, model parameter will increase but not drastically. But the fact that matters is that the Jacobian of this gate would results in a diagonal matrix multiplied by our antisymmetric, the real part of the eigenvalues will be still close to zero due to this

## Simulating RNNs dynamics

|Equilibrium Type|Eigenvalue Type|Behaviour|
|---|---|---|
|Stable node|Real and negative|Stable and attractive|
|Stable focus|Complex conjugate with negative real parts|Stable and spiraling inward|
|Conservative|Purely imaginary|Conservative and oscillatory|
|Unstable node|Real and positive|Unstable and repulsive|
|Unstable focus|Complex conjugate with positive real parts|Unstable and spiraling outwards|
|Saddle point|Real with opposite signs|Saddle point|

Ok so we've seen how theory works behind the dynamics of RNNs, let's see what happens if we use different matrixes for the weights. We can refer to the table we see above. These are the *trajectories* mentioned before!

![[Pasted image 20230609115635.png]]

We can observe in the (g) and (h) pictures that without diffusion the equilibria tend to spiral outwards, instead by using the correct diffusion parameter we can manage to get a cycling pattern. 
Distance from origin for each timestep, we move tangentially increasing the distance from origin lead to instability, this is when $\gamma$ comes in if we choose our diffusion and substract it from diagonal we get a slighly negative eigenvalue that tilts towrd the origing so we have an "opposite" force that lead to constant distance from origin.

These are RNNs that lacks bias and input.

And this is the difference between vanilla and those with feedback.

![[Pasted image 20230609121035.png]]

These are the matrix used for represent the four RNN with feedback (figures e-f)

![[Pasted image 20230609122057.png]]


## Experiments

They tested the long term dependancies of these nets against the LSTM and other models, first experimenting in predicting MNIST digits by pixel by pixel outperforming old model having even less parameters to train, and using $T = 784$ timesteps, then they tried a more difficult task with CIFAR-10, $32\times32$ RGB images in 10 classes, initially with one pixel per channel ($input: m = 3$) with $T = 1024$ steps.

Surprisingly they tried to use an entire row each time of CIFAR-10, after 32 steps the salient information are taken but they used 968 more steps to create artifically longer dependancies from row to row. Still AntisymmetricalRNNs outperformed with less parameters LSTMs

### Is Exploding and Vanishing solved?

Well no, but these architectures indeed mitigate a lot this issue.

![[Pasted image 20230609125212.png]]

Here we can see that regular LSTM the eigenvalues quickly fade to zero indicating vanishign gradient, while the AsymmetricalRNNs are more robust to time steps mantaining the mean close to 1 and stddev close to zero.
But if the values of the $\gamma$ is too big we tend to get vanishing gradient as well.

# Conclusion and Further Works

Basically the problem posed by the paper boils down to find the right initialization for initial value problem such that the 
critical criterion is satisfied.


## References

Nonlinear Dynamics and Chaos by Steven Strogatz.

## Appendix

This is taken from chatgpt 
```ad-abstract

Having all negative eigenvalues in the Jacobian matrix of a dynamical system is a strong indicator of stability, but it does not guarantee convergence in all cases. The behavior of a dynamical system depends on various factors, and while negative eigenvalues suggest stability, additional analysis is necessary to establish convergence.

Convergence refers to the property of a system where its state or trajectory approaches a specific value or set of values over time. The presence of negative eigenvalues indicates that small perturbations around an equilibrium point will decay, suggesting stability of the equilibrium. However, it does not automatically guarantee that the system will converge to that equilibrium.

Other factors can influence convergence, such as the specific form of the system's equations, the initial conditions, and the presence of nonlinearity. Nonlinear systems can exhibit more complex behavior, including limit cycles, chaotic dynamics, or non-convergent behavior, even if the eigenvalues have negative real parts.

To establish convergence, additional analysis techniques, such as Lyapunov stability analysis, are often employed. Lyapunov functions provide a mathematical framework to prove stability and convergence properties of dynamical systems. By constructing a suitable Lyapunov function and analyzing its properties, one can determine whether the system converges to an equilibrium or follows a specific trajectory.

In summary, while negative eigenvalues in the Jacobian matrix are indicative of stability, they do not guarantee convergence in all cases. Additional analysis, such as Lyapunov stability analysis, is typically required to establish convergence properties of a dynamical system.
```

### Linear Algebruh refreshers 

In the context of neural networks and *orthogonal weight matrices*, the *singular values* of the weight matrix indicate the scaling factors associated with the matrix. When the singular values are equal to one, as is the case with orthogonal weight matrices, it means that the weights are rescaled uniformly. This uniform scaling helps preserve the magnitude and direction of gradients during backpropagation, aiding in the avoidance of the vanishing gradient problem

So why stability matters in RNN?

Stability is important in RNNs because it ensures that the network can learn and generalize from input data without becoming unstable or diverging. The instability can occur due to exploding or vanishing gradients, which can make the network unable to converge on a solution. In addition, stable RNNs are better at handling long-term dependencies and maintaining consistency in predictions over time. Stable RNNs also improve the overall performance of the network by reducing errors and increasing accuracy. Therefore, stability is a critical aspect of RNN design that must be carefully considered to ensure effective learning and prediction. 

```ad-attention
The singular values ($\sigma$) of a matrix $A$ are the square roots of the eigenvalues of either $A^T * A$ or $A * A^T$, and can be obtained through the singular value decomposition (SVD) of A:

$A = U * \Sigma * V^T$

where:

- $A$ is the matrix,
- $U$ is an orthogonal matrix $(U^T * U = I)$, representing left singular vectors,
- $\Sigma$ is a diagonal matrix with non-negative singular values $\sigma_1, \sigma_2, ..., \sigma_r$ arranged in descending order,
- $V^T$ is the transpose of an orthogonal matrix $V (V * V^T = I)$, representing right singular vectors.

In this definition, $r$ represents the rank of the matrix $A$, and the diagonal entries $\sigma_1, \sigma_2, ..., \sigma_r$ of $\Sigma$ are the singular values.

```



