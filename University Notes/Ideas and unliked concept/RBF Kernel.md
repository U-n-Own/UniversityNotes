[[Machine Learning]]

## Radial Basis Fuction

Is a [[Kernel]] of defined as follows

$$
\mathcal{K}(x,x'|\tau) = \sigma^2\exp(-\frac{1}{2}(\frac{x-x'}{\mathscr{l}})^2)
$$
Where the hyperparameters are $\tau=\begin{bmatrix} \mathscr{l} \\ \sigma \end{bmatrix}$

### Visualizing RBF

Let's plot each possible similarity as an heatmap

Quite trivial if we're on the diagonal the $x,x'$, so Kernel of two funtion that are near each other are similar... that's right

![[Pasted image 20231130164940.png]]

### What function the kernel produce?

These type: GP is sampling functions with nearby $y$'s for $x$'s deemed similar by the kernel. In other words: you get similar y for similar x

![[Pasted image 20231130165243.png]]

### Hyperparameters

Let's play a bit with the $\mathscr{l}$ parameter: it is the *Lenght scale* and is the differences between the x's

If is small then input pairs have wiggles

![[Pasted image 20231130165746.png]]

For big value instead all pairs are similar and there is less variability in samples

![[Pasted image 20231130165834.png]]

For what regards the $\sigma^2$ is the *Output scale* if we increase it the similarity is more spread out, instead if we descrease the similarity (or decision bound if we use it for SVM kernel) is more strict and sharp

![[Pasted image 20231130170913.png]]

### Little example of sigma action on SVM decision boundary

High sigma could underfit a smaller one could overfit.

![[Pasted image 20231130171353.png]]


