Date: [[2023-03-30]]

Status: #notes

Tags: #ispr [[A.I. Master Degree @Unipi]]

# Gated Recurrent Units and problems

## Dealing with sequences

A RNN must learn an history or **context** that is a more general way to watch to sequential data, they are the simplest form of structured data connected by relationship. This is the simplest form of data because we can build a tree of relationships with more complexity and even graphs.
![[Pasted image 20230330142545.png]]

For example time series: given $c_3$ that's our context given instant $t_3$ the information that comes before that time is the context, the same we can do with tree: given a node the contex is the underlying subtree, with graphs the nodes connected, we would like to capture a very complicated context into a vector of neural activation of the units.

The key of neural network is **Stationarity** using the same set of weights for different length sequences.

![[Pasted image 20230330143415.png]]

Type of RNNs tasks, for example the first is element to element: eg Bracialets that says our current position,
the second one is that we're intrested only in the last output, well if we compute only the error at the end and backpropagate trough time back we're getting gradient exploding or gradient vanishing, an example can be given a sentence we want to know if is positive or negative.
	Item to sequence, giving an item and the item is spitted out as a sequence, well this can be use for image description. The last is basically transduction seq-to-seq like we've seen in [[Lesson 9 - Recurrent Neural Networks#Recurrent]]. 
![[Pasted image 20230330143443.png]]

## Vaniglia Recurrent Neural Networks

$W_{in}$ is used in each of the $h^i$ the same set of parameters unfolded, each of the green thing is a neuron, focussing on the neuron we have that $h_t$ as a vector that take all the output of the neurons at the time step $t-1$. This introduces non-linearity. 
![[Pasted image 20230330144155.png]]

### Forward pass unfolding

If you think about it we're compressing information in all the "history", all the length varying information in a single vector that is our vector of activation of neurons, this encoding is learnt.
![[Pasted image 20230330144515.png]]

### Learning to encode history

Looking at  $h_3$ there will be some history on the preceding steps, if we have a prediction at $h_3$ we can backpropagate (because we can calculate an error), and we can adjust weigths accordingly, going backwards in time, so if $x_1$ was responsible for a failure if the information introduced we can adjust such that the fix will help, this for long dependacies is very difficult.
![[Pasted image 20230330144722.png]]

First issues was described by a master thesist, the long range dependancies.
![[Pasted image 20230330145331.png]]

## Gradients problem discovered


![[Pasted image 20230330145423.png]]

What is really happening to the gradient?
We're summing sum a contribution trough (and for) all the time steps from $t_1$ to $t_{end}$, we've to sum up all contributions. So this means that we have to see how much hidden state at time $k$ contribute to the $t$ step (the last one). The two are directly computable the only one that implies a chain of multiplication is the central.

$$
	h_k = tanh(W[h_{k-1}]...)
$$
This basically
![[Pasted image 20230330150302.png]]
![[Pasted image 20230330145729.png]]


## Bounding the gradient

![[Pasted image 20230330150342.png]]

We can now see this if we explore in gradient magnitude
![[Pasted image 20230330150423.png]]
The product can be controlled by the spectral properties of the Jacobian and the spectral properties of the weights are those who controls the gradient, so if the spectral radious is larger than 1 we will get exploding gradeint and vanishing in the opposite case. So we have to search the problem in the spectral radious of these matrixes, the solution is to search for a design that ensure that both spectral radiouses are equal to 1, but in general this isn't true

## A solution : gradient clipping

![[Pasted image 20230330150754.png]]

We've got weight sharing over time.
![[Pasted image 20230330150911.png]]

## Constant error propagation

How do we solve the problem to get spectral radiouses to 1?

![[Pasted image 20230330151155.png]]

### Activation function
Sigmoid has spectral radious really small, this was the case for CNN that couldn't scale initially, another solution is no activation, because that gives us really a spectral radious equal to 1.  This make a neurons like linear neurons.
![[Pasted image 20230330151409.png]]

Ok now we use only linear neurons how do we choose the weights? We want that our matrices has to be unitary, such that spectral radious keep to stay 1. Ideally this is the perfect solution for our propagation that loose no information, from a spectral perspective this has very nice properties, but seems stupid, well this is basically the core at the LSTMs.
![[Pasted image 20230330151518.png]]

Architectures with complex values activation like modReLU are used also, needs some approximation to make them computationally feasible. First working solution was this simple one, simple identity matrix as recurrency and identify function as activation.

## Accumulator of memory

This is an accumulator, that saturates our memory, we would like to control was goes in the forward pass, and leave the thing flow back with this nice properties, in reality we will take a relaxation of this problem.
![[Pasted image 20230330153001.png]]

## Control reading memory and writing memory aka Mixture of experts

Given an input $x$ each local expert will give a prediction, but then we differentiating weights with respect to the different experts, this is a gating network. Each expert will have an $\alpha$ value that is made like a softmax, in which each of these expert gets a number, and maybe one expert has a vote that is higher. All is parametrized and fully differentiable, so we're learning what expert pick for which region of the space, and each expert is more capable of capturing certain things.
![[Pasted image 20230330153637.png]]

## Forget gate

Error never dissipates, giving hidden states saturation, what was introduced was the *forget gate*: $f_t$ a sigmoid neuron, that decide if we can keep contribution. So selectively picking what we want to get from each neuron, so maybe we want something from a certain neuron, all of another and nothing of another one, effectively filtering, what we do is the *hadamard product* pointwise multiplication between the two vector, remeber that the forget gate and $h_{t-1}$ are the "story" until that moment and $\hat{c}(x_t)$ is the "present".

Well the thing at the right with forget gate has different spectral properties, now that thing has a contractive thing
![[Pasted image 20230330153936.png]]

## Building LSTM cell

This is our recurrent neuron, everytime introducing a new input information, put togheter current input and past activation.
![[Pasted image 20230330154626.png]]

Here we have the *constant error carousel* as the own internal memory, but this is regulated by something else.
![[Pasted image 20230330154724.png]]

We now add another thing
A gating function, we take contribution from $x_t$ and parameters, the ones from the past $h_{t-1}$ sum them to compute a preliminary activation, then we use that new gate, depending on input gate $g_t$ will be allowed to be write (if that is 1), or can be less depending on what is the values.
![[Pasted image 20230330154922.png]]

Let's now add the forget gate
This operates on $c_t$ making such that not all of old $c_{t-1}$ will enter so by filtering. How much of the error carousel continue to flow.
![[Pasted image 20230330155209.png]]

For example if at the output we want to say in a binary string if the zeros are even, we don't want to know about the internal history
![[Pasted image 20230330155348.png]]

## LSTM in equation

Two gates: dependacies of $c_{t-1}$ and one that checks $g_t$. The candidate output then will become the next input, controlling an input will have effect on the input on the next part
![[Pasted image 20230330155455.png]]

# LSTM Design 

Recalling from last time: Gradient vanish is an important problem to solve recalling that memory gets saturated and tends to explode because of positive feedback, we can shield this process with writing and forgetting control gates as we've seen

## Deep LSTM

Using deep short term memories builds a structure like this, with each layer is a layer of recurrent neurons, we can have residual connection and backward connection, for example there are connection that from early layer "informs" the output, this is done at a different scale, in fact in CNN early layer focus on lines, blobs et other small features, more depth layer show respondat to bigger features like faces, the equivalent for RNN is the `sequential resolution` the *time resolution*, layer 1 will likely focus on small time windows and high resolution, the two operate on the output of 1, layer 1 acting as a filter of things with high frequency information and the thirds more abstract and so on. 

Character wise NLP can be seen for the first layer of focussing on syllabs, the second focussing on words and third on sentences.

![[Pasted image 20230412162713.png]]

## Training LSTM

![[Pasted image 20230412163147.png]]

### Dropout

Approach used a lot for regularize, so each time we randomly disconnect units, so two neurons can become the copy of one another by responding to similar stimuly. So for each layer to which we apply dropout, so eachtime a new sample is colledted we sample a binary mask and apply it to the layer the mask will block flow of information between these units, assuring that neurons don't coadapt one another, each batch or minibatch we sample a new, we have an hyperparameter that is the dropout a probabilty to drop the neurons, like we gives usually 0.3, and 30% of neurons gets unactive during the training. Do this during model selection. 

Another thing of dropout is to be used in testing, so we take our sample and run it with a dropout then we do it again and again to get a confidence interval.

We have to use dropout in inference? Yeah but it doesn't help 


In LSTM we do slightly different we do for entire sequence, because if we drop for each item in sequence we are ruining the memory of our network. 

We can also drop connections

### Helping to train with parallelization

![[Pasted image 20230412164710.png]]

Truncated Backprop: Setting a limit of the jumps backwards when we do backprop trough time, because the longer we unfold gradients more we have to do calculation, this is another hyperparameter for model selection.

## Gated Recurrent Units (aka. GRU)

People wanted to reduce parameters and multiplication that brings to GRUs, we have two gates instead of three, so less parameters, but what is the intuition: Using two gates is equivalent to have three!

*Reset* tell how much of the state is computed of recurrent state the $z_t$ performs a convex combination of the two states. The $h_t$ is convex combination of what we received from the past (mixed with the current one), making the constant error carousel disappear, the $r_t$ tells how much of $h_{t-1}$ is used to compute the current state. The reset gate $r_t$ essentially determines how much of the previous hidden state $h_{t-1}$ should be forgotten or discarded, and how much of the new input should be considered for computing the new hidden state $h_t$. This allows the model to selectively forget or remember information from the past, which is particularly useful in processing long sequences of data. Overall, the reset gate helps to mitigate the vanishing gradient problem and improve the performance.
![[Pasted image 20230412164925.png]]

## Bidirectional LSTM for characters recognition

One layer scans the input from a direction and another scans from another
![[Pasted image 20230412165442.png]]

When outputting we fuse from right hand side and left hand side, usually these are used for genetics.

## LSTM for language models

Generic use of LSTM, nowadays we got transformers but with this stucture we can still capture interesting things. Single inputs encode alphabetical characters, then we have layers of LSTMs we're predicting the next character in the sequence, we give to those character by character an entire book, wikipedia and so on. With the last layer having a softmax size of elements of the alphabet. We're predicting a distribution over characters. That is spitted out as 

$$ P(y_{t-1}|x_t,h_t) $$

This is fed back in order to predict the next letter for example we get a `C` then we get `i`.

At training time what we do is teacher forcing: If we input `i` after `C` then we're doing a forced teach to the model, but using this all the time is suboptimal, at beginning of training is useful, then what we do is similar to simulated annealing we start with an high probability of doing teacher forcing then we make the temperature go lower and the probability of teacher forcing is much lower until zero. What happens without is that if we only use teacher forcing if the model gets wrong can't recover because it only used teacher forcing instead if we train it another way.
![[Pasted image 20230412165547.png]]

Whats an intresting thing? 

If we train the network only on jokes we can see that there is a characterization of female characters that changes based on the training data, we can train a maschilistic model. A based model.
![[Pasted image 20230412170649.png]]

At lower level the things are different.
![[Pasted image 20230412170759.png]]

### Recursive Gated Networks

We can see the LSTM as trees unfolding them, this way dependancies are closer, the lenght of linears is $n$ for trees is $log(n)$, especially for depedancy parsing. We're introducing some bias due to this structure the grammar, this way the neural network don't have to learn, people found a way to go from sequential model to trees, the thing with LSTM the assumption on treating a sequence is that we can reorient a sequence, given a certain node in the path we can go backward having a complete ordering, with trees we can do the same but we have a partial ordering this time.

For example if we have a k-ary tree we have the recurrent weights treated this way
Let's say $h_2$ is the parent of $h_4, h_5$.

$$
h_2 = f(W_lh_4+W_rh_5 + W_{in}X_2)
$$

We want to use the same set of parameters for all the node in the trees, we could call the two $W$ up in different ways using $l,r$. So by having 

$$
h_1 = f(W_lh_2+W_rh_3) 
$$

This is called **Positional stationarity** this say that the position in the tree, 

If we use the same $W_c$ for all of the children that we don't matter of the structure of problem or stucture of data.  If we want to reason on what model to use we have to reason on data and basing on the data

![[Pasted image 20230412170945.png]]

## Encoder-Decoder architectures

What if we want to find a compressed representation of recursive neural networks (tree like data).
As soon we compress information on all this data we output a single value, if we want to output a tree that has a different topology of the tree data in output we have to disentangle the representation learnt. And usually in some tasks is impossible, the job of the encoder is to see a structure of an input and given that compressed representation, the decoder will generate a structured output this is how we deal with non vectorial data, for example if we have a generic data and we want to transform them this is ***Transduction*** seen in [[Lesson 11 - Attention and Transformers#Transformers]], these are fully differentiable.

![[Pasted image 20230412172207.png]]

## Captioning images

Use an appropriate encoder for image for example we use CNN to extract information then for decoder we use a LSTM to extract the words like this

![[Pasted image 20230412172953.png]]

## RNN a broader view

These $h_t$ vectors are compressors that keeps a memory like structure in LSTM we've got a RAM like memory.

So by using memory of these RNNs we can recostruct images by using memories stored in them. How do we  save memory? Well if data can be broken into pieces we can rememeber this to do reconstructon for example showing an image of certain pixels, having the pixels in the neighborhood remembered.

![[Pasted image 20230412173303.png]]


### Take home lesson


```ad-summary


```


---
# References

