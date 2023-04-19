Midterm based on these lessons [[Lesson 10 - Sampling Methods]],  [[Lesson 11 - Boltzmann Machines]],  [[Lesson 8 - Markov Random Fields]]

[[2023-03-30]]

#Assignment 

Implement from scratch an RBM and apply it to DSET3. The RBM should be implemented fully by you (both CD-1 training and inference steps) but you are free to use library functions for the rest (e.g. image loading and management, etc.).

1.     Train an RBM with a number of hidden neurons selected by you (single layer) on the MNIST data (use the training set split provided by the website).

2.     Use the trained RBM to encode a selection of test images (e.g. using one per digit type) using the corresponding activation of the hidden neurons.

3.     Reconstruct the original test images from their hidden encoding and confront the reconstructions with the original image (use a suitable quantitative metric to assess the reconstraction quality and also choose few examples to confront visually).

Link to the dataset [here](http://yann.lecun.com/exdb/mnist/)


## RBM procedure

Example of generic RBM structure
![[Pasted image 20230402172316.png]]

The whole system is represented by an energy function $E$

$$
E(v,h) = -v^{T}Wh -v^{T}b - h^{T}c
$$

in statistichal physics high energy configuration are less probable, and the joint probability is intractable because it involves a very large number of particles and their interactions. The Boltzmann distribution describes the probability of finding a system in a particular state, and it shows that higher energy states are less probable. This means that systems tend to be found in low energy configurations, which is consistent with our everyday experience. However, calculating the joint probability of all the possible states of a system is extremely difficult because it requires considering all the possible interactions between particles. This is why statistical physics relies on approximations and simplifications to make predictions about physical systems.

$$
p(v,h) = e^{-E(v,h)}/Z 
$$

What we aim to do is to approximate the likelihood by learning the joint probability distribution over the data, here we use $F(v)$ as the ***free energy*** that is computed this way : $E(v)=-\log p(v)$

$$
p(v) = \sum_{h}{p(v,h)} = e^{-F(v)}/Z 
$$

The idea is to train the model by updating the weights in accord to the Positive step and negative step by derivation seen at lesson [[Lesson 11 - Boltzmann Machines#Derivation]]
$$
\frac{\partial \log p(v)}{\partial w_{ij}}= \sum_h p(\textbf{h}|\textbf{v})v_ih_j -\sum_{v'h} p(\textbf{v'},\textbf{h})v'_ih_j
$$

By doing some simplification the first term is the summation over hidden state vector over conditional probability of hidden states vector given visibles state vector by $v_ih_j$ this is an expectation value, so forgiven the visible vectors those we have, sample the conditionally hidden state vector over probability distribution of this quantity, so an expectation over data, the other component is summing over all states of the RBM (all possible visible and hidden vector) and we take probability of that pair and we "weight" that obervable value, this is an **expectation over the model**

$$
\mathbb{E_\mathbf{data}}[v_ih_j]-
\mathbb{E_\mathbf{model}}[v_ih_j]
$$

These will be used to compute the update of the parameters.

### Contrastive Divergence (CD-1)

Idea:

1.  Replace the expectation by a point estimate at $v'$
2.  Obtain the point $v'$ by Gibbs Sampling
3.  Start sampling chain at $v(t)$

1-step divergence:

-   Positive divergence: $h(v)v^T$
- Negative divergence: $h(v')v'^T$ where $v'$ is reconstructed from a sample from $h(v)$

#### Direct Sampling: first term
![[Pasted image 20230402181301.png]]
#### Gibbs Sampling: second term

This term doesn't  contain conditional probability, so sampling form this is not easy as before, to do this we use Gibbs Sampling.
![[Pasted image 20230402181637.png]]

by doing this we need to Gibbs Sample a lot of time to reach convergence, but with CD-1 algorithm we iterate 1 time
![[Pasted image 20230402181850.png]]
This will provide a slightly better sample estimate, but updating our parameter and doing this CD-1 each time we update the this make work our algorithm.

---

The procedure for training a Restricted Boltzmann Machine (RBM) can be summarized as follows:

1. Initialize the weights and biases of the RBM randomly.
2. Sample a training example from the dataset.
3. Use the positive phase to compute the probabilities of the hidden units given the visible input using the activation function of the RBM.
4. Sample a binary activation pattern for each hidden unit, based on its probability of being activated.
5. Use the negative phase to compute the probabilities of the visible units given the sampled hidden activations using the same activation function.
6. Sample a binary activation pattern for each visible unit, based on its probability of being activated.
7. Compute the difference between positive and negative phases to update weights and biases using stochastic gradient descent (SGD) algorithm with a learning rate $\alpha$:

$$
    \begin{equation}
        \Delta w_{i,j} = \alpha(v_i h_j^+ - v_i h_j^-)
    \end{equation}
 $$
$$
    \begin{equation}
        \Delta b_i = \alpha(v_i^+ - v_i^-)
    \end{equation}
$$
$$	
    \begin{equation}
        \Delta c_j = \alpha(h_j^+ - h_j^-)
    \end{equation}
$$
   
   where $v$ is input data, $h$ is hidden data, $i$ is index for visible units, $j$ is index for hidden units.

8. Repeat steps 2-7 until convergence or some stopping criterion is met.

After training, RBM can be used to generate new samples by starting with an initial set of visible units and iteratively updating them using Gibbs sampling until convergence or some maximum number of iterations is reached.

## Presenting results

To show how our trained boltzmann machines works we can measure the quality of the reconstruction of the images, usually metric used are the mean squared error (MSE) or the peak signal-to-noise ratio (PSNR). These metrics quantify the difference between the input image and the reconstructed image. The lower the MSE and the higher the PSNR, the better the reconstruction quality.



## Intresting connection with hebbian learning

During researches for implementing the RBM i've discovered a lot about different models like the `Ising Model`, these models present some connection with the `Restricted Boltzmann Machines` from the point of view of the temperature in fact both models use the concept of the Boltzmann distribution to model the probability of different states. 

The Ising Model is a mathematical model that is used to describe the behavior of magnetic materials. It consists of a lattice of spins, which can be either up or down, and an energy function that depends on the interaction between neighboring spins. The probability distribution over all possible configurations of spins is given by the Boltzmann distribution.

And then we have `Hopefield nets`, the latter are very intresting because can store *memories* in their weights much like the RBM we've just seen,but with a different approach.

The Hopfield nets are a type of artificial neural network that were developed by John Hopfield in the 1980s. They are often used for pattern recognition and optimization problems. The basic idea behind the Hopfield nets is to use a set of interconnected neurons to store patterns or memories. Each neuron in the network is connected to every other neuron, and the strength of the connection between two neurons is represented by a weight.

To store a pattern in the network, the weights between neurons are adjusted so that when an input pattern is presented to the network, it will settle into a stable state that corresponds to the stored pattern. The network can then be used to recognize similar patterns or to find optimal solutions to optimization problems.

One advantage of Hopfield nets over RBMs is that they are deterministic, meaning that given an input pattern, they will always produce the same output. This makes them easier to interpret and analyze than probabilistic models like RBMs.

Overall, both RBMs and Hopfield nets are powerful models for storing and recognizing patterns in data. Which one you choose depends on your specific needs and goals. 

### Resources

[A practical guide to training RBM by Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)