


## Recap

So after talking extensively about approximating function let's return to see what we're interested

Remeber that we have want to have 

$$
\mathcal{R}(\mathcal{A}(\mathcal{D_n})) -inf_{f:x\to y} \mathcal{R}(f)
$$

And we decomposed this in two main part [[Lesson 1 - Intro to DLT#Risk decomposition]]

For a certain algorithm $\mathcal{A}$ we want to model the optimization error. (By adding the $\hat {\mathcal{R}}$)

We aim to minimize

$$
\hat{\mathcal{R}}(f_\theta)-\hat{\mathcal{R}}(f_\theta*)
$$

We will update our parameter $\theta$ such that our risk will be at each step nearer the true risk, for this we will use gradient descent going the direction of the steepest descent of the function we want to minimize

$$
\theta \longleftarrow \theta - \gamma D_\theta \hat{\mathcal{R}}(\theta)
$$

![[Pasted image 20231218110600.png]]

### Discrete time dynamical system undercover

As we can see these is a type of dynamical system discretized

![[Pasted image 20231218110619.png]]

Evaluating this derivative is really easy and second there are methods that allows to find the zero of our function, but what we want is just to minimize the risk not to find the true minimum (we're approximating function, and not the true unless limits...).

Note: Why computationally efficient?

Computation of $D$(derivative) is quite cheap for NN.

Consider following: Take derivative of a neural net:

$$
D_\theta\hat{\mathcal{R}}(\theta) = 1/n\sum_{j=1}^n D_\theta (f_\theta(x_j), y_j)
\color{red}{D_\theta(f_\theta(x_j))}$$

The red term is the difficult part, the first part is analytical and very easy.

What we do is we store the preactivation of the network.
Let's try to compute the derivative of one of the weights in the net.

So just take the function [[Lesson 3 - Higher dimensional function universal approximation#Multilayer Neural Networks]] and differentiate it.

Computing the gradient of $\hat{\mathcal{R}}$ is cheap

$$
Def : f_\theta(x) = \sigma^1(z(x))w^L\sigma^2(z(x))...\sigma^1(w_k)x
$$

In this case we have evaluation and nonlinearity compoisition this is just *Backpropagation!* 
We start from the last layer and go backward as we can see in the formula.

So computing derivative of an entire neural network is really cheap and convenient to update in this way and why it works.

Usually discrete time dynamical system are hard to deal with, sometimes they are chaotic and you have to carefully choose your learning rate, what's more intresting is continuous time ones.

## Studying the dynamics of Gradient Descent: Gradient Flow

We write formally

![[Pasted image 20231218112529.png]]

As we can see by studying when $\gamma$ goes to zero this seems really like a derivative formulation. We move by steps of length $\gamma$. And this process converging to solution is called *Gradient Flow*. After $k$ timestep of $\theta$ we have progressed to the final solution of our ODE: $\bar{\theta}$

This *Ordinary Differential Equation* is called **Gradient Flow**.

Is just an **euler update** version of gradient descent, if you want to see like that.

In this class we will only use this thing and not classical gradient descent to analyze performances of our networks. 

We need to do some assunption, if the function is too crazy we cannot do nothing.

## Adding constraint to function to be optimized

A differentiable function $\mathcal{R}$ with a set of parameter $\theta$ need to have [[Lipschitzness]] property. Then exists $L > 0$, such that, also called L-smooth.

So the derivative is L-lipshitz smooth iff $D\mathcal{R}(\theta) \leq L-epsilon$

Proof:

![[Pasted image 20231218114715.png]]

Given this we know now that Lipshitz continuity tells us that there exists a solution to our ODE.

```ad-Lemma
Solution to Gradient Flow exists and is unique if $D_\theta\hat{\mathcal{R}}$ is lipshitz
 ```


 ```ad-Lemma
 If we assume Lipshitz continuity then for every $T > 0$ there exists $c > 0$ such that, the evolution of the ODE

$$
||\bar\theta_{k\gamma} - \theta_k|| \leq C\gamma
$$
```

```ad-Proof
![[Pasted image 20231218115303.png]]
```

