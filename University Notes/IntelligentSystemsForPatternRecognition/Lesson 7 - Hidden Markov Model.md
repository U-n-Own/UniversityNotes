Date: [[2023-03-08]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Part 1

Dynamic Bayesian Networks are like counterpart of RNNs in the probabilistic world, this will be two lecture.
What happens if RV are latent and we want to work with them? How do we do inference in this model?
Well there is a concept of ***message passing***, one can define backprop in terms of message passing, so local exchange of information that serves to obtain a global effect. In probabilistic model this process is used to implement efficient inference. 

- Sum-product msg passing
- Max-product msg passing

Using then inference to learn : *Expected maximization* algorithm discussed in [[Lesson 5 - Generative and Graphical Models]]

## Sequences

Like we did with time series we can have probability time series called sequences, so obervation at specific time $t$ depends on the previous observation, in word of sequences cannot be considered independent, but you must consider the history

![[Pasted image 20230308162019.png]]

We have a reference population as set of i.i.d. sequences $\textbf{y}^{1,}..., \textbf{y}^N$

## Markov Chain and HMM
![[Pasted image 20230308165145.png]]
![[Pasted image 20230308164811.png]]
![[Pasted image 20230308164838.png]]
![[Pasted image 20230308164853.png]]
![[Pasted image 20230308164914.png]]

We can generalize to more complicated $nth$ order markov chains but the matrix will grew a lot in dimension! 
Also now we've seen a Markov chain of thing we knew. And this was for letters!

Now let's do with `words`, we get a size of english vocabulary, a lot bigger than alphabet. 
![[Pasted image 20230308165415.png]]

Does this make sense? 
We've to model the transition probability of all the possible combination of syntactical, because relationship and dependancies between words are semantic dependancies, not grammar, so we've to model the fact that the article *A* is followed by a noun *cat*. This information are hidden. Now with markov chain we would like to capture this behaviour of language, *language induction*.
![[Pasted image 20230308165641.png]]

Those superclasses: **Article, Noun, Verb** are unknown (unobserved). We're decoupling the two part of the models dependancy and part of 'word emitting'. In this specific setting what the hidden state are doing is to assign an unobservable symble to a certain subsequence in the current time, so a clustering operation. You're assigning a *discrete* value to those, given what you've seen so far. **Clustering of a subsequence**.
There can exist modeled with continues values: Infinite transition Markov Models.

The **Prior** distribution is a Multinomial.

![[Pasted image 20230308170347.png]]

If $y_t$ is discrete that distribution is multinomial, ($y_t$ rows in the matrix, like word in the vocabulary) times $S_t$ 
hidden values that are $C$.
For non discrete we have that the emission distribution is a gaussian distribution, a mixture of gaussians, how many? $C$ gaussians where C is the number of $s_{t}$ hidden states.

![[Pasted image 20230308170744.png]]

The $\textbf{Y}$ is the joint of $y_1..y_T$, so we're summing over all possible assignment we can give to all possible hidden in all the timesteps. Very big summation. Now we can apply conditional probability assumption.
Factorizing $P(S_1)$ that doesn't depend on anything (prior). Then for all the other time steps doesn't matter what's happen in $S_{t+1}$ if we are observing $S_t$, this because the previous d-separate [[Lesson 6 - Conditional Independence and Causality#D-separation]] (path blocks) for the one after.  

## Recursively encoding

What we do is recursivly encoding the history of your input information in the activation of the recurrent layer. RNNs are representented differenly but (some specific HMMs) do the same things.
![[Pasted image 20230308172217.png]]

Like unfolding RNNs we can unfold these models, also there are connection with Automata.
Self loops says that we stay there with some probability $p$, or emitting to a different states. These can be generalized to transducers.
![[Pasted image 20230308172511.png]]

### 3 Notable inference problems

Now that we have these models we can do inference on those.

***Smoothing*** Inferring a posterior, given something that is observable, what will be the probability that tells us
***Learning*** Inferential problem of given a set of observed sequences (data), we want to find *prior* $\pi$, $A,B$. How do we find them? Solve a maximum likelyhood problem.
***Optimal state Assignment*** Find optimal assignment of all the state given a visible sequence, so we have a possible sequence. Solved by max-product message passing (*Viterbi algorithm*)

![[Pasted image 20230308172651.png]]

## Forward-Backward Algorithm
![[Pasted image 20230314094005.png]]

How do we **exploit factorization?**

In this case, $P(S_t = i | y)$ represents the probability of being in state $i$ at time $t$ given the vector of observations $y$.

According to Bayes' theorem, this conditional probability can be written as:

$$P(S_t = i | y) = P(S_t = i, y) / P(y)$$

where $P(y)$ is the probability of observing the vector $y$.

Now, if we assume that the state i at time t is conditionally independent of all other states and observations given the state i-1 at time t-1 (which is a common assumption in many probabilistic models), we can rewrite the joint probability as:

$$
P(S_t = i, y) = P(y | S_t = i) * P(S_t = i | S_{t-1})
$$

where $P(y | S_t = i)$ is the probability of observing the vector y given that the system is in state $i$ at time $t$, and $P(S_t = i | S_{t-1})$ is the transition probability from state $i-1$ at time $t-1$ to state $i$ at time $t$.

Substituting this into Bayes' theorem, we get:

$$
P(S_t = i | y) = P(y | S_t = i) * P(S_t = i | S_{t-1}) / P(y)
$$

Now, since we are only interested in comparing probabilities, we can ignore the denominator $P(y)$, which is constant for all values of i. This gives us:

$$
P(S_t = i | y) ≈ P(y | S_t = i) * P(S_t = i | S_{t-1})
$$

This equation tells us that the conditional probability of being in state i at time t given the observations y is approximately equal to the product of two probabilities: the probability of observing y given that the system is in state i at time t, and the transition probability from state i-1 at time t-1 to state i at time t.

Thus, we can say that P(S_t = i | y) is approximately equal to P(S_t = i, y), which is the joint probability of being in state i at time t and observing the vector y.

We want to derive a solution for problem 1.
![[Pasted image 20230308232853.png]]

Since the blue doesn't depend on the red part 

$$
P(Y_{t+1:T} | S_{t}) \\ \; \color{blue}\beta_t(i)
$$
$$
P(S_{t} | Y_{1:t}) \\ \; \color{red}\alpha_t(i)
$$

We decompose the problem of finding the posterior in another of finding these two probabilities and multiply one depends on the past the other on the future.

This is called **alpha-beta** recursion, one start from the future and goes back to the past the other way around. I want now to write $\alpha$ and $\beta$ is to write them in terms of *transition, prior, emission*.

Now we apply conditional independance and manipulate those probabilities to evaluate in a way to get them under terms we know.

$$
P(S_{t}| S_{t-1})
$$
$$
P(S_1)
$$
$$
P(Y_{t} | S_t)
$$

These are what we know.


We want to compute

$$
\alpha_{t{(i)}}=P(S_{t=i,}Y_{1:t})  = \sum_{j=1}^{C}P(S_{t}=i, S_{t-1}=j, Y_{1:t})
$$

How can we rewrite this? Think to red rectangle i want to separate $S_t$ from the rest, what RV i can use to do it? $S_{t-1}$, but how can i introduce it? By Marginalization as we can see up.
Now we can apply conditional independance assumption. So what we need to bring out? Well $Y_t$!

If we do this we can see that $S_t$ d-separates $Y_t$ from the previous ones

So we get : This is the **Emission distribution**

$$
P(Y_{t}| S_{t})
$$


.
.
.

-- Minute 44:00 lesson 

So for the $\alpha$ get a vector and every element in the joint probability computed recursivily and they passing each in time this vector. 


## Sum product message passing

We've message travelling in two directiong $\alpha$ and $\beta$, these when meet each other can compute a probability. This $\Psi$ term measurer the affinity between the term $n$ and $n-1$ summing over all the instantiation of our RV : $C$ instantiations.

An example would be : we have in usual msg passing counting how many $a_s$ we have instead hidden state i and j and we "count" how many times we find a j after i something like that.3

![[Pasted image 20230309143615.png]]

# Part 2

We're still on the problem of learning some parameters, that will be fin $\pi, A , B = \theta$ by maximum likelyhood
![[Pasted image 20230309144217.png]]

We start from those we know $Y^n$ then we apply the factorization of the observable with respect to the markovian distribution so *prior*... 
Introducing by marignalization the hidden (the summation).
Problem is that we've log of a product and into there we have a summation, that is not that good, since we wrap our parameters into a log of summation... Expectation Maximization comes in help.

We introduce **Indicator variables**
$$
z_{ti}^{n}= 
\begin{cases} 1 \;\text{if n-th chain is in state t at time i}  \\
 \\0
\end{cases}
$$

We can transform the summation in there to a multiplication by elevating it to the indicator variables, so a multipication of something to $0$ that is $1$ with a certin probability.
Then we have a product over time and over the $ith$ and $jth$.
![[Pasted image 20230309145048.png]]

Then our outer log pass between all the productory and they becomes simply summations, the indicator variables becomes and go out. The major problem is that we derived this we need to know $\mathbf{Z}$, but thank to expected maximization we need just the $\mathbb{E}(\mathbf{Z})$
![[Pasted image 20230309145513.png]]


How do i compute the $Q$ function? you can get the Expected values that runs of target variables
$Z | X, \theta^{(t)}$ consider $Z$ as variables and the rest is constant. Expectation of constant is constant so they doesn't matter so much. Then we use the function at previous step to get the new parameters, we don't need to know the actual values but just the expectation.

## Graphically

We're searching for the top of red curve project it to the blue curve and repeat this until we get to the maximum of the blue curve. This is going to give us a non decreasing trajectory. So we have a model of non-lesser likelyhood. ***Elbow*** in ML : the red one is a lower bound.
![[Pasted image 20230309145822.png]]

### E-step

How do we compute the expectation of the function upward? (Summation one)


Try as an exercise this derivation
![[Pasted image 20230309153656.png]]


## HMM at work


In 2008 we have an high value of **volatility**, so we can use HMM to a time series to try to model it. What we're seeing in red the actual probability and in black the HMM prediction. In this case we have low and medium volatility if we increase the number of states we can capture more market regimes.
![[Pasted image 20230309154013.png]]

Example with more states
![[Pasted image 20230309154237.png]]


## Decoding problem

This is about finding the optimal state assignment of a state $s = s_{1}^{*},...,s_{t}^{*}$ for an observed sequence $\textbf{y}$ 

There is no unique interpretation of the problem.


## Viterbi algorithm

An efficient dynamic programming algorithm based on *backward-forward recursion*, running first backward then forward. This is an example of max product message passing algorithm, because we take the maximum of all possible states, if we aren't careful state blows up in number!

We're trying to solve this problem of maximizing: Joining maximization

So we're writing $\epsilon$ backwards than after we choose the max at that time depending on time 2 we... *complete this part*

Remind from last time, we're doing a max product message passing essentially.

## Input-Output HMM


![[Pasted image 20230314111701.png]]

Your input information are integrated inside the $S_t$, so that is a recurrent layer that capture information from the hidden state at precedent steps.
We're in a discrete latent space

### Bi-directional HMM

Sometime the order of things doesn't matter, if the sequences unfolds in time, you can't go from back to start, but for proteins encoding, makes sense, for example if you read a protein in the two way.
![[Pasted image 20230314112016.png]]


These two different markov chain learn to generate some output by fusing the information by left and right side, you can have genomic sequence in input. And you want to get the binding protein from the two coming direction.

### Coupled HMM

Let's say we have two interacting processes like respiration and EEG, they have different time dynamic, so you capture the time dynamic in one latent space (the first chain), and then the other, if you want to use them you can couple the two information. Maybe the two chains have information on different things but have a fair shared of "feature". In this coupled space they are affine to each other.

Useful for multimodal data for example

![[Pasted image 20230314112147.png]]

## Dynamic Bayesian Networks
![[Pasted image 20230314112504.png]]

Think to graphical model that want to find ancestral tree for someone, these model unfolds but not in time in some structure like we can see above, the model expanded for the number of nodes a children can have. Time is one possible of structural network, that model sequences, if they are acyclic you can model those with bayesian networks.


### Take home lesson

- HMM clusterizes subsequences into latent state assignment. Basically mixture model (mixture of gaussian depending on each other)
- Inference in HMM: 
	- forward backward: to get the posterior
	- Expecatation maximiztion for parameters learning
	- Viterbi for most likely hidden state sequence (maximization of likelyhood)

```ad-summary


```


---
# References

