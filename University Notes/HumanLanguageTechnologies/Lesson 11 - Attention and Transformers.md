Date: [[2023-03-28]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]]



## Sequence to Sequence with NMT

Neural Machine Translation works well, using an encoding RNNs for input and Decoder RNNs for output.

When all the useful information goes into the last hidden state of the input and we are translation from the decoder we have a **bottleneck of information** at the last step of input. 

How to solve it?


# Attention

Provides a solution for this problem of bottleneck, the core idea is to *watch to the most influential and important word at that time in that specific sequence*, attention is a collection of weights that produces a vector of weight giving more importance to some of the words. We have a scores for attention for each of them to see how these work.

What attention provide:


## Attention behind machine translation

Sequence to sequence are good for transduction (seq-to-seq), these have limits, these are sequential so we have to go one by one and we have bottleneck.

The ***Transformers*** doesn't use recurrent network or convolution operation, just relies on attention mechanism or (self-attention), we move from a sequential to a parallelized model that speedsup the things.

Since we want to parallelize and RNNs are sequential we have to do something about this
![[Pasted image 20230328093627.png]]
So LSTM, RNNs needs attention, can we just rely on attention and get rid of recurrent? This allows to parallelization, the answer is:

![[Pasted image 20230328093748.png]]
This above is the Transformer architecture encoder-decoder but allow to do it in parallel


### Attention definition

There are two way of seeing attention:

- Querying: a vector of query, that is build on these questions so which values are we focusing on, computing the weighted sum and chosing the most *important* item in sequence

Intutition:
- The weighted sum is a *Selective summary*, and attention is a way to obtain a fixed-size representation of an arbitrary set of representations.

#### Formally

We have values $v_1,...v,_n$ and some keys $k_{1},..., k_{n}$

Attention involves :

1. Compute attention scores : $e \in \mathbb{R}^N$
2. Taking softmax to get attention distribution : $\alpha$
3. Using attention distribution to take weighted sum of values obtaining an attention output called *context vector*

#### Variants of attention

- Basic dot-product attention: used in NMT
- Multiplicative attention: introduce a matrix that multiply query and keys for this matrix
- Additive attention

#### Issues?

- With recurrent models

RNNs unroll left to right this encode linear locality that is a useful heuristic, also nearby words are often good for the context, problem in RNN is that takes $O(seq\;length)$ steps for distant word pair to interact reducing effect of information trough the net. So long distance dependancies.

Also we can't rely on sequential model to exploit fast parallel computations, so training on large corpus of data the cost grows linearly with the dataset dimension and this inhibits our model training speed.

- With CNN

Unparalleliziable, losing long distance dependancies, using windows convolution taking ngrams that slides trough the phrase that give local information and still has no solve long dependancies..

- With attention

Treating each word representation as a query to acces and incorporate set of values, a good thing is **Self-Attention**, so applying attention to itself.

## Self-Attention

This can be tought like a query using hashtables!


Standard Attention

![[Pasted image 20230328095551.png]]

---

Self-Attention

![[Pasted image 20230328095619.png]]

So we can think of that as a memory.

In **self-attention** core of Transformers is taking attention to itself

1. Transform each word embedding with weight matrices $Q,K,V$, $\in \mathbb{R}^{d\times d}$, where these are queries, keys and values
2. Compute pairwise similaties between keys and queries and normalize with softmax

$$
e_{ij}= q_i^Tk_j\;\;\;a_{ij}=\frac{e^{e_{ij}}}{\sum\limits_{j'}e^{e_{ij'}}}
$$

3. Compute the results 

## Vector notation for self attention

With matrix operation we can better understand so

1. Embeddings stacked in $X$, calculate our matrixes
$$
Q=W_QX\;\;\;\;\;K=W_KX\;\;\;\;V=W_VX
$$

2. Calculate attention query and keys: $E=Q^TK$
3. Add softmax to this to get a distribution : $A = softmax(E)$
4. Output is finally the weighted sum of values: $Output = AV$

Or
$$
Output = softmax(Q^TK)V
$$


## Self attention in NLP

With self attention we can replace recurrent so stacking LSTM layers stacking self attention, but it has some issues.

First self attention is operation on sets, so there's no notion of order.

We need some changes first

*First*: Solution to these problem are adding sequence index as a vector $p_{i} \in \mathbb{R}^d$, for $i \in {1...T}$

But how is done this vector?

### Position repreentation with sinusoids

The vector is concatenate sinusoida![[Pasted image 20230328102730.png]]l functions of varying periods, giving us an index in sequence made this way
![[Pasted image 20230328102730.png]]
- Pros: periodicity indicate that aboslute position isn't important and we can extrapolate longer sequences as period restart
- Cons : Not learnable

*Second* thing, how do we get non linearity, since attention is linear, we have to introduce non linearity afte computation of attention, we add a FeedForward network to post process each output vector

$$
m_{i}= MLP(output_i)+b_{2}= W_{2}ReLU(W_{1}\times output)...
$$

*Third*: we don't have to peek at the future when doing NMT tasks so we use a mask to don't lookup at certain future words that should not be seen, this allows still paralle discarding portion we don't want to see. 
![[Pasted image 20230328103239.png]]

These three things ensures that for Neural Machine Translation this work, but what if we generalize a model to a new architecture?

# Transformers
![[Pasted image 20230328103454.png]]

The encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$$ to a sequence of continuous representations $z = (z_1, ..., z_n)$
Given z, the decoder then generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time.
At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

Generally a transformer is made like this but stacks many of these layers
![[Pasted image 20230328103610.png]]

Like this
![[Pasted image 20230328103639.png]]

## Encoder

Simply our encoder is like this
![[Pasted image 20230328103702.png]]

## Encoder-decoder
![[Pasted image 20230328103735.png]]

Putting them togheter we get a layer.

The encoder’s inputs first flow through a self-attention layer – a layer that helps the encoder look at other words in the input sentence as it encodes a specific word.
The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position.
The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence

## Self-attention in image

What really self-attention is doing represent a set of weights that looking at the weights on a certain word will describe how the rest of the phrases relate, for example when we see the 'it', stronger weights are connected to 'The', 'animal' and this was referring to 'it' as pronoun to the animal. Solving the issue of anafora.
![[Pasted image 20230328103849.png]]

### In details

We're multipling each query for the $W_Q$ matrix and so for the keys and values
![[Pasted image 20230328104236.png]]

The score is get with multipliying using dot product with $q_i$ and $k_i$ then we have a normalization by some scale like divide by 8, and finally applying softmax


## Multi head attention

We have multiple matrix of attention, like BERT that has 12 attention layer so 12 keys-matrices.

What we're going to learn actually is the matrixes $\{W_i^Q,W_i^K,W_i^V,...\}$ for all the attention heads $i$
![[Pasted image 20230328104618.png]]
Just applying different attentions to each layer, where each attention is focussing on different things like this, more than one attention matrix for each layer
![[Pasted image 20230328105053.png]]

## More on Transformer architecture

For the forward pass we have simply
![[Pasted image 20230328105136.png]]

![[Pasted image 20230328105150.png]]

### Tricks: training with residual connection

This is useful is basically taking shortcuts, so by passing the smoother function we help a lot to navigate in the space during training, the gradient of identity is easy is 1 for example, so we can propagate the initial input to make it easier to learn and this is yet another method for keeping under control the 
![[Pasted image 20230328105227.png]]

### Tricks: Layer normalization

This is very good to get faster training where the Idea is: cut down on uninformative variation in hidden vector values by normalizing  to unit mean (m) and standard deviation (s) within each layer.

Goal is to ensure that distribution of activation of previous layer isn't too much shifted from the other, since all the layer are the same in this case.
![[Pasted image 20230328105413.png]]

Simply subtract the mean and divide by the variance.
![[Pasted image 20230329145930.png]]
### Trick : Scaled dot product attention

Here we're adding a scaling towards the $\frac{Q\cdot K^T}{\color{yellow}{Scale\;factor}}$ if the lengh of our embedding is $d_{k}$ then the scale factor will be $\frac{1}{\sqrt{d_k}}$

![[Pasted image 20230328105622.png]]

## Summary

This thing can be repeated more than once stacking these as layers
![[Pasted image 20230328105708.png]]

## Why self attention?

We can se that using self attention do consistenly better than other models, but also improve performance and speed

![[Pasted image 20230328105927.png]]

## Multi head attention revisited

So what are we doing is working with something that has another dimension axis, these are layer stacker over to build powerful models that learns more than 1 type of attention, for example we have a set of three attention matrixes at the end, we can perform this calculation still parallellized.
![[Pasted image 20230329144557.png]]

And then at the end we combine all the attentions
![[Pasted image 20230329145058.png]]
Then the $Z$ is forwarded to the next layer and so on.

## Transformer encoder

![[Pasted image 20230329150055.png]]

## Transformer Decoder
![[Pasted image 20230329150339.png]]
## Cross attention

Cross attention is the flow of attention from encoder to decoder
![[Pasted image 20230329150454.png]]

On top of the decoder whe have a linear layer and a softmax, this provide a score for each possible words in a vocabulary

![[Pasted image 20230329150818.png]]

## Extracting output

So *linear layer* at the end take the vector produced by decoders and merge it into a logits vector.
Then softmax spits out probabilities.

![[Pasted image 20230329152520.png]]


Results in data when attention were introduced.

![[Pasted image 20230329152935.png]]

### Document generation

We can directly generate entier documents
![[Pasted image 20230329153126.png]]

### GLUE collection of NLP tasks


![[Pasted image 20230329153258.png]]

## Bigger is better?

If we just provide more data, the system just becames better and will beable to do things that before couldn't do. Maybe this wasn't even trained to do certain things, and there are no trace of this grow that stops.
![[Pasted image 20230329153924.png]]

# Transformers architectures

Now we have the building blocks and we can start to construct things, so what are the current best variances of these architectures?

- Start with pre-trained word embeddings with no context 
- Learn how to incorporate context in LSTM while training on the task

Issues:
- Training data we have must be sufficient but usually are much smaller and must be sufficient  


## Pre-trained whole model

In modern NLP all parameters are initialized by pre-training these methods hide part of input on the model, and then use this to learn certain subtask, so we can build certain complex transformers then we use those pre-trained and fine tune them with few data to make model that are adapt to certain specific tasks.

Why pre training works? 

### Stochastic gradient descent and pretrain or finetune

Pretrain language model provide a base parameter $\hat{\theta}$, finetuning a model on task initializing this parameter help because the gradient descent goes relatively close to our parameter during finetuning, maybe finetuning, goes in an optima that is better for our taks but not for another, clearly an example of no free launch theorem, we become better at soemthign esle


 
---

Possible use of transformers : next time only using decoders, encoders or both togheter!

---
# References

[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

