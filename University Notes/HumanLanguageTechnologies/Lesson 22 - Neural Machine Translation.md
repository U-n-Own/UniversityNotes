Date: [[2023-05-10]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]],[[Lesson 11 - Attention and Transformers]], [[Recurrent Neural Network]],[[Lesson 9 - Recurrent Neural Networks]]

# From statical methods to Neural

First were introduced sequential models to translated word by word so encoder-decoder architecture over RNNs. We have an **Encoder** architecture and **Decoder**.

![[Pasted image 20230510145018.png]]

Then how do we train these NMT systems? Well we still need parallel corpus!

## Training NMT

We then minimize the cumulative sum of errors in time, classic backpropagation trought time.
But this architecture has no intermediate or extra steps all boils to matrix multiplications, and encoding into vectors our representation of words.

![[Pasted image 20230510145647.png]]

Being obtained this probability distribution at each step, what do we do? We can do the simple approach : *Greedy decoding*, taking the most probable prediction for our model and then do it for each step. But there is a problem, sometimes if we choose everytime the best we cannot "explore" the space of possible answers, if we get in a greedy decoding with an error we can't go "back" because of the fact that current answers are based on the history, if we do it greedy we can't escape. 

We could explore all the possible translations and pick the best one, but this is computationally expensive.

![[Pasted image 20230510150127.png]]

### Beam search decoding

Using beam search we restrict a lot the space of possible solutions. 

![[Pasted image 20230510150256.png]]

Example of beam search

We have each time 4 alternatives and we drop two of them to process toward the tree we would like to obtain
A problem here is where to stop, so choosing a stopping criterion is hard.

![[Pasted image 20230510150455.png]]

## NMT do not solve all the hard stuff

![[Pasted image 20230510152452.png]]

Some of these problems can be solved by just using more data.

The big jump in NMT was because of [[Lesson 11 - Attention and Transformers]] and [[Lesson 16 - Advanced RNN#Attention in equations]]

![[Pasted image 20230510153215.png]]

>[!info]
> 






---
# References

