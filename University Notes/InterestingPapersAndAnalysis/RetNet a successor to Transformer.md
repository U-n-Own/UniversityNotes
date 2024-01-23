[[Machine Learning]], [[LLMs]], [[Parallel Computing]], [[Recurrent Neural Network]]

**RetNet** paradygm is a new proposal for an architecture that can solve the problem that transformer has, we need to keep informations about the **Attention Matrix**, given the context of our net.

![[Pasted image 20231005181238.png|center]]

## The essence of RetNet

Linearize all the stuff, so why RNN cannot do parallel stuff?

We need to process the token from the start and then propagate the signal, if we want to train, we need forward and then backward propagate the signal. We do 1 token at a time.

In Transformer with the introduction of Attention, everything can attend at the same time at everything. Eg [[Causal Mask]], we can only see backward information, so every token becomes a training example


$$ax+by+cz={SomethingCrazy}$$
	This is a linear thing, what we would like to do is to treat it recurrently and acting parallel on the computation. That's the aim of RetNet 

- $ax \rightarrow \gamma$
- $by+\gamma \rightarrow \gamma$
- $cz+\gamma \rightarrow \gamma$ = SomethingCrazy

This is the recurrent formulation, we're wrapping - unwrapping the computation, so this requires to do inner calculation and then go out, in order and *sequentially*.

Basically recall how [[Lesson 14 - RNN Sequential Models and Gated RNN#LSTM in equation]] was defined...

How do we parallelize this?

$$
\mathbf{v} = \begin{bmatrix}
a \\
b \\
c
\end{bmatrix}
\quad \text{and} \quad
\mathbf{u} = \begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
$$


$$\mathbf{v}^T\mathbf{u} = SomethingCrazy$$

## Why Transformers aren't Recurrent?

Softmax is the main problem


$$softmax(\mathbf x)_i = \frac{exp(x_i)}{\sum_{j=1}^n exp(x_j)}$$
Well softmax in order to go sequential must compute one by one all the $exp(x_i)$ needs to be computed, then we aggregate them togheter, it can be done, but the tradeoff is too demanding.

What we do? Let's *yeet* the Softmax away.

So do this work? We're getting rid of a big representation power that softmax give us.

## Retention mechanism

Dual form of recurrence and parallelism: Train in parallel, infer in recurrent fashion.

We start from Recurrence and go in Parallel world, more details in the [paper](https://arxiv.org/pdf/2307.08621.pdf)
For what i understood they project the input in a one dimensional function $v(n) = X_n \mathbf{w}_V$, then from the recurrent formulation they take a matrix $A$, go full monke (diagonalize) eigendecomposition of this matrix

### Retention formula

$$
Retention(X) = (QK^T \odot D)V
$$
Where $D$ is our causal mask, blocking the view to the future tokens, this is done in a decaying fashion as we can see.

This scalar will decay by going back in time: the further back in time the more the effect will be less useful... 

Also $D$ contains the [[Lesson 11 - Attention and Transformers#Position repreentation with sinusoids]]

![[Pasted image 20231005185204.png]]

*D be like:*

![[Pasted image 20231005185101.png]]

So (old) Attention was something like this : [[Lesson 11 - Attention and Transformers#Vector notation for self attention]]

### Recurrently writing Attention

![[Pasted image 20231005185839.png]]

![[Pasted image 20231005185927.png]]