Date: [[2023-03-27]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]]

# Idea behind LSTM

Central idea is to **Learn to forget**

Our cell in LSTMs are more sophisticated than vanilla RNNs

Cells stores long term information to carry over, taking info from previous states output selecting which information can pass, this cell work like a sophisticated RAM. Some of the data are stored in a clever way.


## Gates

We've three different gates that control flow of information, each gate control how much the info flow of the $h_{n-1}$ step given we are in step $n$. 

- Forget gates: decide what to forget from previous (irrelevant information on *current input*)
- Store gates: decide what to keep (relevant information on *current input*)
- Output returns a filter information flow on the current (Uses tanh or ReLU)

![[Pasted image 20230327143528.png]]

In vanilla RNNs instead we have only one gate, that joint the previous with current signal.

## Solving Vanishing gradients

LSTM the information can survive longer: more than 100 time steps instead of Vanilla RNNs can mantain information on about 7 states before incurring in vanishing gradient. After 2013 LSTM were very good at achieve SOTA results, now we have ***Transformers*** [2018-now]. With the introduction of Attention.

How do we solve gradient vanish in LSTM:







# GRU

Another variant of RNNs that is simpler than LSTMs, GatedRecurrentUnits, these have a similar structure but simplified because combines the *forget* and *update* gate in a single gate. Also merging cell state and hidden state in a single one.


## Vanishing in GRUs

The prolem in RNNs emerges into the part of updating $h_t$

$$
h_{t}= f(W^{hh}h_{t-1}+W^{hx}x_t)
$$

The error is propaganting along all the nodes in the path

### Generalizing to Feedforward vs Recurrent

The difference in RNNs is the fundamental that the matrix of weights is repeatdly used between all the states all is accumulated in this single matrix, in feedforward model that want to sequential stuff we need a lot of different matrixed and is a waste of memory.

# Attention mechanism in Machine Translation

We're doing Many-to-Many RNNs, we can extract Attention information trough an LSTM for each word in a sentence and compute weights that understand what are the most relevant information at the current time step. This was used to do machine translation to understand what are the most important words to be translated in a certain part of the text.

So we use the attention weights to understand what word to place in translation at given step in sequence.


## Sequence to sequence chat model

Transformers model actually follow a model of LSTM encoder-decoder stage.

So for a chat  model we have an input that is taken by an encoder and a decoder that spits out an output on the context (gathered by LSTM)

## Image captioning using attention

With CNN we can work on image and plug attention here what happens is that we can further describe our image, so making caption on images for example, and we can also see what is the important stuff for the model to infer the informations.

Last year we had great newcomes like DALL-E and Stable Diffusion combining images and text with the use of attention mechanism.

Tomorrow transformers, how they are a breaktrough in NLP, Vision and many more!


>[!info]
> 






---
# References

