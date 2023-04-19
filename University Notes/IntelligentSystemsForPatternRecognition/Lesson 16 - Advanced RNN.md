Date: [[2023-04-13]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Dealing with structured data

 Two lectures today and tomorrow

![[Pasted image 20230413142459.png]]

## Limitations of RNN (GRNN)

In recurrent models we get weights sharing that is very good because the parameter are smaller, they are all the same in fact they are used in the next sequence and also backprop in time, in the [[Lesson 14 - Seqeuential Models and Gated RNN#Encoder-Decoder architectures]] we've seen compositional pieces like encoder decoder.

There are piece relevant in a certain time, other pieces that are not relevant and pieces always relevant

![[Pasted image 20230413142712.png]]

Attention is what deal with this `relevant` wording.

# Sequence to sequence and attention


In genral two sequence are not syncronized words appear in different order, and many word can be expressed by 1 or more word in another language. 

![[Pasted image 20230413142953.png]]

What is the context, since with RNNs used to compress context, so the context is a compression of information in the first path, the simplest approach is using a RNN to model seq-to-seq, this way as we can see, we're compressing in the context $x_n$ the encoding of the final word, that depends on the encoding on the previous words. Then the model has a tag like $<TRANSLATE>$ token, so from then on we have to start translating the input sequence until we get a $<EOF>$ token. These architectures don't work because of the context lenght, we're mixing two languages, if we ask to modify the vector as modifying trough the decoding, we have to keep track the output we're generating at the same time, this is not scalable. 

![[Pasted image 20230413143205.png]]


We need encoder-decoder scheme
![[Pasted image 20230413143744.png]]

Is the originally $c = h_n$ good enough? No, we now need another recurrent neural network separated from that, that has a different tasks, because the latter has a compression task, the next needs to extrapolate information and spit it.
![[Pasted image 20230413143936.png]]

This one is another RNN, receiving in input a context vector $c$, we can decide to assume $c = s_1$, but if we don't want to lose memory we can as well do $W_c$ a separate set of parameters that are connected to all the other, then the decoder unfolds, this is a generative use of a recurrent neural networks

![[Pasted image 20230413144202.png]]


The $y_1$ is our ground truth and we can use teacher forcing at training time at step $s_i$ we have that function that has context, sequence and the ground truth.
![[Pasted image 20230413144254.png]]

### Sequence to sequence learning 

Overall structure we can see this encoder-decoder.
![[Pasted image 20230413144425.png]]


If we reverse the order of the input sequence the model works better. Why?

If we see : "The cat is on the table" -> "Il gatto è sul tavolo".

Doing this make words near each other : "table the on is cat the" -> "Il gatto è sul tavolo"

## Pay more attention on decoding

![[Pasted image 20230413144937.png]]

Then what we can do, the attention module gets information from the encoder

![[Pasted image 20230413145031.png]]

## Black box

We have a module for each of the $h_i$, we want an aggregated context that depends on the input but also on the information $S$ that is the context but that has to change as the context change

![[Pasted image 20230413145043.png]]

## Let's take a look inside the box

![[Pasted image 20230413145223.png]]


## Box

![[Pasted image 20230413145240.png]]

### Relevance

### Softmax

### Voting

There is this component that we can think about like a scaled dot product or MLP, given two vectors in input compute a scale that give us how they are aligned these then are parametrized to learn good measures, for each of them we're gonna compute the $e_i$, one for each of $n$ inputs, then we get these numbers that tells how much each of the $h_i$ are **relevant** for the current context, then we just apply a **Softmax** over those and we get a probability distribution, then we can perform a convex combination of the $\alpha_i$. (**Voting**)

![[Pasted image 20230413145300.png]]

### Attention in equations

![[Pasted image 20230413145748.png]]

Here we've got a vector of recurrent neural activation, then we will have something giving us context on which we will generate the $C$. What is the assumptionThe assumption is that the context provided by the input data will help generate the appropriate memory cell $C$ for the given sequence. 

## Focussing on attentional module

![[Pasted image 20230413150017.png]]

Assume we're decoding at step $s_3$, so the full input is being fed, we generated hidden state for each word in input sequence, the $s_i$ comes from the decoder, we want to generate $y_3$. We will use the previous $s_i$ as a new context for the next prediction that becuase input for $s_3$ then we compute $y_3$.

## Attention visualized
![[Pasted image 20230413151900.png]]

Attention was used also as an explainer of what neural network is learning but is questionable what is really learning sometime.


## Seq-to-Seq on steroids

![[Pasted image 20230413152128.png]]

## Hard attention
Instead of a combination of all the $h_i$ we take one, problem is that random sampling is not differentiable operation, backpropagating there what cause? The gradient should pass from there. We can do pass trough (just go and i don't mind really), policy gradient, and other.
![[Pasted image 20230413152325.png]]

# Transformers 

[[Lesson 11 - Attention and Transformers]] We've seen it in the other couse of HLT, this architecture encoder-decoder, we have attentional layer in the decoder and in the encoder.

![[Pasted image 20230413152554.png]]

## Self attention

Q: Explain attention and self attention main differences, where we use self attention and where simple attention in transformer (endoer-decoder) architecture

*Attention* is a mechanism used in transformer architectures for natural language processing tasks such as machine translation, language modeling, and text generation. Attention **allow the model to focus on certain parts of the input sequence while processing it.**

Simple attention is a type of attention mechanism where the **model computes a weighted sum of all input tokens at each decoding step**. The weights are computed based on their relevance to the current decoding step.

*Self-attention*, also known as intra-attention, is a type of attention mechanism where the **model attends to different positions within the same input sequence**. It allows the model to encode information about each token based on its own context and its relationship with other tokens in the input sequence.

In transformer architectures, self-attention is used in both the encoder and decoder layers to capture contextual information about each token in the input sequence. Simple attention is used in the decoder layer to attend to specific parts of the encoder output while generating an output sequence.

In summary, self-attention allows for capturing contextual information within a single sequence, while simple attention focuses on specific parts of different sequences. Self-attention is mainly used in transformer architectures for encoding information from an input sequence, while simple attention is mainly used in decoder layers for generating outputs from encoded information. 

Computing attention between each of the inputs $X_i$, but before we want to transform these
![[Pasted image 20230413152807.png]]

### Self attention with examples
![[Pasted image 20230413153015.png]]
![[Pasted image 20230413153243.png]]
![[Pasted image 20230413153437.png]]
![[Pasted image 20230413153454.png]]
And finally

![[Pasted image 20230413153529.png]]

## Visualizing multi head attention 

![[Pasted image 20230413153656.png]]

## Multi self attention heads in equations

![[Pasted image 20230413154008.png]]

## Transformers are not sequential

If we want to model sequences, to make them work on sequential data where position matter we add an information in the encoding into the embedding an additional embedding that is the position of the words, stupid way is to use a vector one hot encoded but this has many problems because we need to encode position on a very large vector, a better way is to use $\sin, \cos$. For even and odd positions.

This under is a sentence with 50 tokns, taking the first slice orizontally is the encoding of the position zero, each time we go down one level we're sliding like bit shifting. We can transform one position to another by using **rotational matrixes**.
![[Pasted image 20230413154538.png]]

## Attention in computer vision

We're doing the small features in low level pixel by pixel (treating them like word that compose an image from a CNN).
![[Pasted image 20230413155430.png]]

## Vision transformers

![[Pasted image 20230413155534.png]]

Self attention is the key, key-query is very good, while the rest of architecture stays the same.



### Take home lesson

Well use Attention, is efficient all is parallelizable, the good trick of self attention, Soft attention is nice because is fully differentiable hard attention is stochasti and cannot backprop in theory.
Both of them are sentive to different things.

Encoder and decoder scheme is a composable architecture 



```ad-summary


```


---
# References

