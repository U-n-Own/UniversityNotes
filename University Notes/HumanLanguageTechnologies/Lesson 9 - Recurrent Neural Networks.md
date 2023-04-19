Date: [[2023-03-22]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]]

# Language modeling recap

What we will see is how language models will be used in the case of sentences.
Language model computes probabilities of each sentences in a language

$$
P(w_n | w_{1}, w_{2},..., w_{n-1})
$$

The model that compute this probability is a *Language model*


## Markov Assumption

We a have seen how with Markov Assumption, so saying that $x_n$ depends only on the $x_{n-1}$ preceding words

![[Pasted image 20230322143006.png]]


major problems with n-grams models are *sparsity*, *storage* and the fact that with growing n sparsity worsen.

## Neural language models

Neural language are being successful, by having no sparsity problems and not storing all the past n-grams, we just store a vector for the words, so we have a big complex vector that contains all words information.
Remaining problems are the too small fixed windows and if we enlarge the models becomes difficult to train $W$ gets bigger. The matrix weight $\textbf{W}$. We've also asymmetry for each word in sentences because each $x_i$ is multiplied by a different weights in $W$. We want an architecture that can process any length input. Repeatdly apply the same network over different inputs that are correlated like words.
![[Pasted image 20230322143529.png]]


## Recurrent

- RNNs are called recurrent because they perform the same task for every element of a sequence with output depending on the previous. 
- RNNs have memory which capture informations about past calculations
- RNNs can make use of an arbitrarily distant information, so long sequences.

Here is a simple RNN diagram, we can we that we can unfold it in time, but not only in 'time', what these things really do is unfold in sequences.
![[Pasted image 20230322143953.png]]

At each stage we have two part that comes into play
The hidden state output is passed over and each words is worked one step at a time, at the end we can see that we're getting the same distrubution
![[Pasted image 20230322144439.png]]

## Hidden units

Hidden state are the memory of the network, this rememeber the information that are useful for compute the next state, at the end we have the output computed at the end, given the whole sequence of words.

The recurrent point is that we also use less weights and then we use them repeatedly to work on the entire sequences, reducing the parameter to learn!

RNN have advantages like

- Any length input
- Model size not increasing with output
- Computation use information at previous steps
- Weights sharing in timesteps and shared representations, reusing and updating the same weights in accord to the entire structure so working with word in any position in the sentences changes how they are updated

and Disadvantages

- These RNNs are slow, if we had fixed and forward computation this is recurrent.
- In principle the information from long time steps is lost over steps

## Reducing complexity

![[Pasted image 20230322145321.png]]

## Deep RNNs

What happen if we get more layer in a RNN? We could add a layer up and another over it.![[Pasted image 20230322145526.png]]
This is what happens.

Another intresting thing is flowing information from both sides right to left and left to right these are called Bidirectional RNNs
![[Pasted image 20230322145623.png]]
These are powerful for speech recognition and NLP because if we know the next words and store some of the previous context we can be better at next prediction

## Training RNNs

![[Pasted image 20230322150039.png]]
We're using the logarithm because if there is not loss we get 1, our $T$ is the number of tokens in the dataset then we average the loss in the training set. But this would take too much time, so what you typically do is pick batches of sentences and repeatedly train for those, really like Stochastic Gradient Descent.

## RNN Tagger

For our model we will see in the [example](https://medialab.di.unipi.it:8000/user/v.gargano1@studenti.unipi.it/lab/tree/HLT/Lectures/RNN-Tagger.ipynb) we're gonna use a Gated Recurrent Unit that is made like this
![[Pasted image 20230322152229.png]]

As usually we use softmax as output because we want to spit out a distribution probability.

## Backprop trough time

Calcolate crossentropy then backpropagate that have to go all the way trough the unfolded network that's the step the cost the most in the training trought the time.
![[Pasted image 20230322153228.png]]
This leads to serious problem, because we're multipling a lot of gradients, doing this we have a step that if we have a single layer network but unfolder trough a lot of time steps the *Vanishing gradients* problem if very serious!

Long sentences have these type of problem if we have to bring some keypoint in a long period the information is lost after a while. Later we will se that with ***attention*** this mechanism is solved that's what use now LLMs. 
So large and Deep RNNs struggle with this point, there a way to mitgate this problem, but we cannot solve, we can put a threshold that make that stop or use ReLU instead of Sigmoid, more techniques we've seen in [[Machine Learning]] are gradient clipping or simply reduce the learning rate.

### BPTT From book and MIT lessons

- Forward pass trough time
- Backprop trough time
We have to backpropagate error individually across all time steps individually *and* then across all the time backward, so errors backpropagate in time to the start of our sequence
![[Pasted image 20230327173145.png]]

But what happens if the flow of information is big?
![[Pasted image 20230327173453.png]]
Well as we can see the computation is really big and gets big very fast because of the many matrix multiplication trough time. 

If gradients values tend to be larger than 1, these repeated multiplication make the gradient explode, while for smaller values make it disappear, some trick that work for exploding one are *Gradient clipping*  

## Examples of RNNs usages

Problems are very serious but if we use them for sentences to sentences or language generation is still doable
![[Pasted image 20230322154520.png]]
Or also
![[Pasted image 20230322154604.png]]

## LSTMs

In LSTM the flow of information the core idea is the state $C_t$ that changes slowly with linear interactions. It's very easy for information to flow along it unchanged.


>[!info]
> 






---
# References

