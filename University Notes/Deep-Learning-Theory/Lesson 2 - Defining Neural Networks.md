[[Machine Learning]],[[Theoretical Machine Learning]]

# Population risk decomposition

Last time we formalized the problem that we want to solve, we want approximate a $P$ probability distribution, $l$ our loss function, a function class (parameterized) $\mathcal{F_\theta}$ and a class of algorithm that will minimize the risk : $\mathcal{A}$.

Remember from the last time we have : [[Lesson 1 - Intro to DLT#Summary]], we want to study what happens when number of data goes to infinity.

The first step of this decomposition is

$$
\mathcal{R(f_\hat\theta)-\inf_{f:X\to Y}\mathcal{R}(f) = \text{Empirical error?(review)+Approximation error}}
$$

Approximation error depending only on the function class we can represent.

## What is an (artificial) Neural Net?

What is the corresponding $\mathcal{F_\theta}$

```ad-abstract
A neural net is a class of function that combines non-linearity with 'nets' of linear transformations
```

Different choice of non-linearity and transformation will define different class of function, today we will see *Shallow* NN, so single layer.

## Def. Single Layer NN:

```ad-important
Let $\mathcal{X} \subseteq \mathbb{R}^d$. A single layer NN of width $m \in \mathbb{N}$

Is a function of the form 

$$
f_\theta(x) = \alpha \sum_{j=1}^m \alpha_j \sigma(w_j.x+b_j)
$$
```

- Where
- - $\alpha \in \mathbb{R}$ is a scaling parameter of NN
- - $a_j \in \mathbb{R}$ are the output weights, but we can also have a *vector* as output
- - $w_j$ is a *vector* are input weights
- - $b_j$ are the biases
- - $\sigma: \mathbb{R}\to\mathbb{R}$ is the activation function or nonlinearty

We will define our $a = (a_d, w_d, b_d)^m \in \mathbb{R}^{(1+d+1)m}$

```ad-note
Most of the time the output dimension will be set to $d' = 1$
```

### Graph neural net representation

![[Pasted image 20231211110619.png]]

This is an intresting class of function we will analyze.

Often, in this lesson for example, our *scaling factor* will be $\alpha=1$. 
Using matrix multiplication, define $\sigma: \mathbb{R}^m \to \mathbb{R}^m$ componentwise

$$
f_\theta(x) = \alpha \mathbf{a}^. \sigma(W\mathbf{x}+\mathbf{b})
$$

Where bold are vector and W is a matrix $\mathbb{R}^{m\times d}$

### Different activations functions

The main classes of activation function are *ReLU*, *Sigmoid*, *Tanh*...

For example we can define the $ReLU$

$$
\sigma(z) = z_+ = max(0,z) = z \mathbb{1}(z \geq0)
$$

Another widely used is *Sigmoid*.

But we're free to choose arbitrary activation function, that may work not well in practice, in fact there exist a lot of variations of those like *Leaky ReLU* that is not differentiable at zero (we just trick by define it's derivative zero), also we have saturations on those we maybe will talk later.

A widely recognized of smoothed ReLU is $\sigma(z) = z \Phi(z)$ where the $\Phi$ is a gaussian

We can also use use *Gaussian PDF*: $\sigma(z) = e^-{z^2}$ and the commonly square function $\sigma(z) = z^2$, the latter we will see it's not a good choice.

![[Pasted image 20231211112228.png]]

## Function classes 

Fixing an activation function we define a 

$$
\mathcal{F_{\sigma,m}} = \{\alpha\sum_j^ma_j(w_jx+b_j):(a_j,w_j,b_j \in \mathbb{R}^{1+d+1} \forall{j} \in \{1,m\})\}
$$

and $\mathcal{F_\sigma} =$ all the possible functions defined above

#### Example : Let $X \subset \mathbb{R}$

$$
PL_m = \{f \in C(\mathcal{X}. f(x)= (\tilde a_hx+\tilde c_h)\mathbb{1}(x \in [\tilde b_h, \tilde b_{h+1}]\}
$$

Where the $\tilde b \in \mathcal{X}$ and $\tilde a, \tilde c \in \mathbb{R}$

#### Lemma 2.1

For every $h \in PL_n$ there exists a $f \in \mathcal{F_{ReLU, m+1}}$ such that $f(x) = h(x) \forall x \in \mathcal{X}$ (So $PL_n \subseteq \mathcal{F}_{relu}$

```ad-important
Proof: Wlog let $h_\to^{x\to -\infty} 0$

Set $w_j = 1, \forall j \in {1..m}$ 

$$
f_\theta(x) = \sum_j^{m+1} a_j(x+b_j)\mathbb{1}(x+b_j \geq 0)
$$

by distributing the $j_s$

$$
\sum_j^m(a_jx + b_ja_j)\mathbb{1}(x \geq -b_j)
$$

So for $h = 1, -b_1 = \tilde b_1, a_1 = \tilde a_1$
and for $h \to h+1$ we need to choose a new slope that will fix the cumulated slope before, where cumulated slope until that moment in defined by $\hat a_{h+1} = \sum_j^h a_j + a_{h+1}$. (Cumulated slope + new slope that ajust such that is the one that approximate the current slope)
```



We want to approximate each interval by using a new neuron, so we have a neuron for each "breakpoint" in our function.
### Graphical idea:
![[Pasted image 20231211114617.png]]


Now we have proven how big this class (with the relu), we can approximate any **continuous piecewise linear functions**, what can we say about any other function? 

## Approximating error for single layer NN

We would like to study if we can approximate any function


Def: a class of function is an universal approximator if for we can approximate any **continuous function over a compact set $\mathcal{X}$**
if for every $g \in C(\mathcal{X})$ there exists $f \in \mathcal{F}$ with $|f-g|_\infty \leq \epsilon$. 
We've seen that we need a number of neurons that grows if we want to approximate well

#### Prop 2.2: 

Let $g: \mathbb{R}\in[0,1] \to \mathbb{R}$ first let's take one dimension be a $g-lipshitz$ then $\forall \epsilon > 0 \; \exists f \in \mathcal{F}_{ReLU,m}$ with $m=ceiling\{{g\over\epsilon}\}$ such that $|f-g|_\infty \leq \epsilon$

Proof: By [[Lesson 2 - Defining Neural Networks#Lemma 2.1]]: We only need to show $\exists h \in PL_m$ with $||g-h||_\infty \leq \epsilon$.

Given this following lipshitz function, we divide the function in parts and then we interpolate

![[Pasted image 20231211120531.png]]

![[Pasted image 20231211120850.png]]

By assumption we can say that the $\tilde a_j \leq \epsilon\rho$

Then we have that $|g(x) - h(x)| \leq^\triangle |g(x)-g(b_j)|+|g(b_j)-h(x)|+|h(b_j)-h(x)|$
By triangle inequality we show that is lower bounded

This lemma gives us an intuition behind the expressiveness, and this by only considering ReLU

#### Lemma 2.2

```ad-hint
From real analysis we know that : Piecewiese continuous linear function are dense in the space of continuous functions on compact sets
```


Let $\mathcal{X}$ be compact subset of $\mathbb{R}$, then $PL=\bigcup_{m=1}^\infty PL_m$ is dense in $C(\mathcal{X})$

Proof from real analysis: (Rudin) see the book

Why linear continuous function are dense in 

##### Cor 2.3 $F_{ReLU}$ is universal for every $\mathcal{X} \subseteq \mathbb{R}$


### Exercise: 

```ad-todo
Prove that $\mathcal{F_{sigmoid}}$ is universal in $\mathcal{X} \subseteq \mathbb{R}$
```

Idea: By corollary 2.3 $\exists f_m \in \mathcal{F}_{{relu},m} s.t. \: ||g-f||\leq$$\epsilon\over 2$, $\forall g \in C(\mathcal{X}), \exists m$

![[Pasted image 20231211122353.png]]

So what we do is to approximate the ReLU with a sigmoid


#### Lemma 2.4:

Let $\mathcal{X} \in \mathbb{R}$ compact, $\mathcal{F}_{\mathbb{1}}$ is universal on $\mathcal{X}$

Proof:

## Non linearity that do not universally approximate:

Necessary condition is that our set $\mathcal{X}$ has to be compact.

*Polynomials* (of bounded degrees) are close under linear transformation so we cannot construct universal approximators.

These are not dense in $C(\mathcal{X})$ for fixed $h$

Also *Identity* 

```ad-note
Remark: $\mathcal{X}$ needs to be compact in general : $sin(x)$ :$||sin(x)-f(x)|| \geq 1 \forall f \in \mathcal{F}_{ReLU}$
```

Easily we can see that we need an infinitely wide neural network to approximate this.

### Why we're using infinite norm?

Eg

$$
\int ||f-h||^2P(dx) \leq ||f-h||^2 \int P(dx)
$$


## Hints for next lesson

Until now we working in single dimension, but we would like to go to higher and *infinity* dimensions, for example we could build blocks in $\mathbb{R}^2$ in order to approximate our functions as we did now and how do we write this as a function of 1-dimensional elements?

eg: $\mathbb{1}(x \in B_j)$

Single layer neural nets can only add non linearity they cannot multiply non linearity, but since multiplication is a repeated multiplication well... just add more neurons
