

What happens when neural network is infinitely wide is the thing we will discuss.
Today we will continue from what we left last time

How expressive are neural networks we defined as

```ad-summary
A class of function $\mathcal{F}$ is a universal approximator of $\mathcal{X}$ if  $\forall g \in C(X)$ we have $\exists f \in \mathcal{F} ||f-g||_\infty \leq \epsilon$, with $\epsilon > 0$
```

### One dimensional

```ad-Corollary
For every $X \subseteq \mathbb{R}$, single layer NN, with ReLU activation are universal. (Since we can approximate ReLU with other function the proof work for other activation too... well there are cases with quadratic not counting.)
```

```ad-Remark
With X being compact, for not compact X just think to approximating cosine with a ReLU, to a certain point we cannot approximate well...
```
### Reason about infinity norm

We use infinity norm because we're bounding on an integral: bounding approximator and the function in the *Risk function*, integral with the not know probability measure.
# Approximating in higher dimension

Topic of today is: ***can we get much higher?***

Q: Can we set a $d$ such that we can approx $\mathcal{X} \in \mathbb{R}^d$?

One idea can be use step function, since these are dense in the set of continuous function we know that we can approximate.

So using *Indicator functions*: Approximate $\mathbb{1} (x \in B_j)$: $B_d - X_{j=1}^d[b_{j,k}, b_{j+1,k}]$

Can i use a product of those function? How do i do with neural nets? 

Like this : $\prod_l \mathbb{1}(x\in A_l)$ with $A_l = X_{j=1}^d[b_{l,d}, b'_{l,d}]$

With this $f = \sum a_j \sigma(w_j x+b_j)$

We want to perform a product but our network has a sum in there (the sigma can be our step function).

## How to represent a product with a sum?

Indicator function are difficult to be represented as product, but if we use other non linearities we can do something.

Recall that we can use trigonometrical function

Consider :
$$\sigma(z) = \cos(z)$$ 
We can write this as *Algebra Condition* we will use later

$$
\sigma(z)\sigma(y) = cos(z)cos(y) = cos(z+y) + cos(z-y) = \sigma(z+y) + \sigma(z-y)
$$

So this trigonometric identity say that we can represent a product with a sum of linear combinations, but that's what a single layer NN is. 

How do we use this thing for our purpose?

We want to represent products, so our nonlinearity want to represent product as sums! So our class of function is closed under product. Hoping this is enough for approximate in higher dimensions.

Try to approximate any continuous function $g \in C(X)$, and leveraging on the Fourier Transform we know that cosines can approximate any function of a given dimensional space! With linear superposition.

Then we approximate each one of those cosines with an arbitrary wide single layer NN and we're done!

Let's see how it's done

![[Pasted image 20231217171317.png]]

The last passage is leveraging on the expressivity of the $\sigma$ function

## Stone-Weierstrass approximation

This approximation theorem is not very different and says a sufficient condition for a class of function to be dense in the set of continuous function.

```ad-important
Theorem (Stone-Weierstrass) : Let $\mathcal{F} \subseteq C(X)$ for compact sets in $\mathbb{R}^d$ satisfy

A) For every $x \in X$ there exists $f \in \mathcal{F}$ such that $f(x) \neq 0$

B) For every pair $x,x' \in X$ with $x \neq x'$ there exists $f \in \mathcal{F}$ with $f(x) \neq f(x')$ ($\mathcal{F}$ separates points)

C) $\mathcal{F}$ is closed under pointwise multiplication (Is an algebra)

Then $\mathcal{F}$ is an universal approximator
```

The first part says that is not zero everywhere and the condition B and C are exactly the condition we saw before, the B is expressivity condition

Let's check if our neural network class satisfy these conditions.

```ad-Lemma
$\mathcal{F}_{cos}$ is universal
```

```ad-Proof
Check for SW conditions

Cosines are continues.

A: $cos(0x) = 1 \;\; \forall x$

B: $x \neq x' \to f(x) = cos(\frac{(z-x')(x-x')}{||x-x'||^2})$ satisfies 
$\begin{cases}
f(x') = 1 \\
f(x) = 0
\end{cases}
$

C: we already checked this.
```

Cosine activated neural networks are universal.

As corollary of this lemma we can extend to neural network that use cosine to any network that has sigmoid

Suppose $\sigma$ a continuous non linearity is sigmoidal: so for the limit to infinity we gets to 1 and limit to minus infinity gets to zero

Then $\mathcal{F}_{\sigma}$ is universal and the same goes for ReLU

![[Pasted image 20231217181022.png]]

The $h_u$ is a linear superposition of 1-dimensional functions (cosines) combining the high dimensional object to a 1 dimensional, so now we want to approximate well these function one by one...
So our $h_{u,j}$ is a single neuron mapping: $\mathbb{R} \to \mathbb{R}$ and we showed last time that we can approximate that function and then we do one for each of those, since those are continuous in one dimension. This show that for every non linearity neural nets are approximator also in higher dimensions.

Ther are non-linearity that do not cover these conditions eg paper from Leshno 93': $\mathcal{F_{\sigma}}$ is universal iff $\sigma$ isn't polynomial.

# Multilayer Neural Networks

Definition: Let $L \in \mathbb{N}$ a fully connected feedforward neural network with ($n_1, ..., n_L \in \mathbb{N}^L$) is a function of the form

$$
f_\theta(x) = \sigma_{L+1}(z^{L+1}(x))
$$
Where **preactivations** (vector of quantities of network messages that arrives at each neuron and of which non linearity the neuron act)

(So preactivations are the z that are obtained when the signal is propagates and so before applying the non linearity of the next layer. Basically the linear combination of what comes from the previous layer)

These preactivations are given by :

$$
z_d^l(x) = \alpha_{l-1}\sum_{j=1}^{kl-1}w_{jk}^l \sigma_l(z_k^{l-1}(x)+b_j^l)
$$

And the first is defined as

$$
	z_j^1(x) = \sum_h^d w_{jh \times h}^1+b_j^1
$$

![[Pasted image 20231217190139.png]]

## Graph notations

![[Pasted image 20231217190154.png]]

## Deep Neural Networks are universal

Taking ReLU as activation.

Proof

Since for $z \in \mathbb{R}$ i can represent the identity $\mathbb{1}(z)$ = $-(-z)_+ + (z)_+$ = $-1\sigma((-1)(z))_+1\sigma(1z)$

We can construct a depth $L$ neural network combining $L-1$ layers of identity with the network from the 2.7 Theorem

![[Pasted image 20231217192923.png]]

What this theorem says is : take a neural network with a depth of order of $L^2$ and a polynomial in number of weights then if you take an insufficiently deep network so with a square root over the $L$ much shallower, then you need at least an exponential number of parameter $2^L$ to get to the same part