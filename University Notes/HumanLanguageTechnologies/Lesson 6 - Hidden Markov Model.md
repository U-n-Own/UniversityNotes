Date: [[2023-03-14]]y

Status: #notes

Tags: [[Probability]], [[A.I. Master Degree @Unipi]]

# Markov Chain

We spoke extensively of this models in [[Lesson 7 - Hidden Markov Model#Markov Chain and HMM]] this lesson.

A **first order observable Markov Model** aka Markov Chain consist in a set of states $q_1,...,q_{n}\in Q$ 
And a set of transition probabilities $A_{ij} =a_{01}, a_{11}...$  Where $a_{ij}$ is the probability transition of state $i \to j$.  Of course these probabilities has to be summed to 1, because each state has to be a probability distribution.

## Hidden Markov Model

Essentially this is a Markov Chain in which we can see only the output symbles, that are not the transition, that we don't know and we will treat these as latent RV.

### Speech recognition HMM

What we want to discover is the probability of these transition

![[Pasted image 20230314102528.png]]

The three main problems we will see are the same we saw in the other course in this section [[Lesson 7 - Hidden Markov Model#3 Notable inference problems]]

First problem can be solved by *computing total likelyhood of a sequence* but this is rather expensive given that it grows a lot : $O(N^T)$ with $N$ hidden states and $T$ observations.

### Speeding up with Forward Backward

We've seen this algorithm recursive in [[Lesson 7 - Hidden Markov Model#Forward-Backward Algorithm]]

Essentially we can visualize it like this

![[Pasted image 20230314104212.png]]

### Viterbi algorithm and decoding

We want
![[Pasted image 20230314105544.png]]

What are we doing? We go forward taking the probabilities and then we backtrack on the thick lines.
So compute max value in nodes and then backtrace to get the maximum
![[Pasted image 20230314105352.png]]

## Learning probabilities: Baun Welch algorithm (Expectation Maximization)

![[Pasted image 20230314105726.png]]

We can use this algorithm for part of speech tagging
![[Pasted image 20230314105806.png]]

![[Pasted image 20230314105914.png]]

How can we formulate this problem? 

![[Pasted image 20230314110038.png]]

We're casting our problem into an HMM. For example the problem of transition to one tag to another can be modelized by this, we just count in the text the number of times there is a transition and divide by the number of those tag in the text

Example

![[Pasted image 20230314110346.png]]

The model disambigued "race", it can be verb or a noun, the probability in this case is higher for verb.





>[!info]
> 






---
# References

