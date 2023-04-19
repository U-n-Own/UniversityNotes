Date: [[2023-04-14]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Optimizing memory: Neural reasoners 

Let's start with hirarchical and multiscale Recurrent Networks, like a working memory for next predictions, we talked about gated RNN claiming they can solve a problem in long term dependancies and vanishing gradients, learning to solve complex tasks that require storage and recall of information is not achievable.

Some architectures that capture long range sequences

## Skipping state update mitigate vanishing gradients

For a error at $h_3$ we need to backprop from there to the beginning for each time step (for which we update states). What happens if we don't update the state? Basically we're doing a skip trough connection, we can go from $h_0$ to $h_3$ for example, backpropagating with less vanish. Not updating all the states. How can we skip updates?

We add skip connection from a certain unit $A$ to a far unit not the next one for example. (Residual connection)
![[Pasted image 20230414142217.png]]

Other solutions are to insert inside the architecture : not using recurrence, but using convolutions. The first CNN was used on time series, so we have a seq-to-seqence model

What's good with convolution? We can have a size of kernel that is the size of context or memory, $k$ is small typically but we can stack convolutional layers, to get a bigger context.

Given a set of inputs $\{x_1,x_2,x_3,x_4,x_5,...\}$ if we have a kernel of $k=3$, capturing $x_1,x_2,x_3$, but we can stuck convolutional layer that a single part of same $k=3$ takes the kernel and also these are parallelizable, not sequential so very fast to compute, but we're still far from very long term dependancies.
![[Pasted image 20230414143045.png]]

## Vanishing gradient depends on the shallowness

From transformers paper we can see that complexity per layer (sequential operations) and path lenght (how much need to propagate in network).
As we can see self-attention is quite expensive.  The three architectures have different behaviours.

So we have 

- **Self-attention:** with sequential operation $O(1)$ since each element in the sequence can attend to all other elements in the sequence independently, without requiring any sequential operation. And maximum path lenght describing the longest path between any two elements in the sequence is also $O(1)$, as every element can attend to any other element directly without going through any intermediary elements. This makes self-attention highly efficient and scalable, allowing it to be used effectively on very long sequences.

Additionally, self-attention is also highly parallelizable, as the attention scores can be computed independently for each element in the sequence. 

- **Recurrent**: instead is really sequential in fact operation are computed in $O(n)$ time complexity, where $n$ is the length of the input sequence and maximum path length in the network of $O(n)$ because each step in the sequence depends on the previous step, forming a recurrence relation. 

- **Convolutional**: Pay inted in complexity per layer since we have the kernel that participate to the multiplication.  In this case sequential operation and maximum path lenght are $O(1)$ and $O(log_k(n))$ because each layer only involves a fixed number of operations and the number of layers is proportional to the logarithm of the size of the input. This makes convolutional neural networks very efficient for processing images and other high-dimensional data.


![[Pasted image 20230414143611.png]]

## Hierarchical RNNs

How do we introduce skips connections:

- Static skipping: hard coding for example skip every 3 elements.
- Adaptive skip: learn when to skip

We can adjust the granularity of the skip: units, block or entire layers.

### Regulariation : Zoneout

![[Pasted image 20230414144311.png]]

At each timestep **Zoneout** draws a mask and if the element in the mask is 1 then it's updated the, this thing isn't differentiable since we're random sampling what to update, but we can't do it and seems to work (not very elegant).

Charateristic of this is stochasticity, so losing differentiability, but this thing act as a regularizer, isn't even much different from dropout.

## Clockword networks

Since zoneout is random chance we can thing to a static way approach to design skipping connections. This shitload of arrows is composed of parts these hidden layers, and we can thing that each of those $T_1,T_2,...$ these operates at a different clock, updating states at different frequencies.

For example $T_g$ has an update clock of 1 every 10 seconds, if we have a data stream of $1 \;Hz$  this is the "slowest updated". The module at left is one that runs at a different frequency, the one at left need smaller or larger frequency? Well higher frequency because we want to transfer high frequency information to the next module, so the first capture very fast changing details in our sequences and the ones at right part becomes thing that capture things in a long range of dependacies
![[Pasted image 20230414144745.png]]

## Skip RNNs

What's happening here is that the bottom line here $S_{t-1}$ is the history while the $\tilde{u}_t$ is used to manage if update or not. If we do not update just copy the last state. This is a parametrized neuron that will learn when to update or not.
![[Pasted image 20230414145418.png]]

### Skip models on MNIST

We're treating it like a sequence problem seeing the pixels one by one. Some pixels are being ignore other are attended.
![[Pasted image 20230414145702.png]]


## Making network efficient by exploiting hierarchical structure

Take a NLP sentence we want to analyze, a document, there structure is evident, at the lowest level we watch to character and goin upper in level we see words and setences, at higher level paragraphs, then agglomeration of paragraphs, and so on. This sequence is not flat, we can create a hierarchy structure.

And we can trigger one of those operations based on what we're considering.

![[Pasted image 20230414150206.png]]

Flush : send update in the next layer and zero everything, why? if we finish a word for example "the cat is on", in letters level we have to flush when finishing a words for example: **t,h,e_,i,s_o,n...**

Also these are a lot of non differentable operation: Copy and flush. For update we can use pass trough and other.

### Recap : 1

Each of these structure has a different approach.
![[Pasted image 20230414150610.png]]

### Recap : 2

And techniques used to get a larger context of "long range dependancies".
![[Pasted image 20230414150658.png]]

# Neural Reasoning

Solving more "biological" things, reasoning is quite complex, planning even more, chaning the state of the algorithm during the execution (internal evolution) and spitting output when the computation concludes, introducing Kleene operator in the neural networks.

Recurrent models are our general programs, neural networks are turing equivalent one of the condition is that has to be recurrent, to iterate, so we need recurrency. We want to inbue a neural reasoning by example, old school algorithms wants preconditions, we can give noisy or missing data to neural networks, so running an algorithm inside a neural nets is quite ok.

### Basics of RNN and memory

![[Pasted image 20230414152426.png]]

For taks 15 we need to understand that gertrude is a particular type of animal and is afraid of another.
These are complex question answering that can be solved with neural networks.

Powerful linguistic model can be solved of nowadays can solve these trough prompting for example, but we're doing a different approach.

We need a memory in which to store facts and retrive them in a very specific way in order to do reasoning. Facts are meshed with others (cocktail party problem), we need memory that stores thing not togheter and is able to retreive.
**External memory**.

## Memory networks

We've a memory network that acts as controller or processors (like computer processors retreive things to memory then writes them and so on). Then we have a memory model, separated like Von Neumann computer model.

![[Pasted image 20230414152810.png]]

This Network has to learn some funtions

Feature maps (adaptive) encoding information with embedding layer, then we can use measures of similarity. Once is encoded we can write a piece of it, we can overwrite another fact, when we study we get things wrong when we discover that we're doing wrong we want to overridden wrong information to write the new one discarding old.
![[Pasted image 20230414152951.png]]

## End to end memory networks

Reasoning is end to end, not the memory operations!

Every line in the emedding is a list of fact of things we know these are stored here, that is a memory or determines how the memory is composed, memory controller can only reason on those.
Since we're doing question answering problems fact are stored, we get in input a question in natura language, we encode it in some way, obtaining an embedding matrix, something differentiable, then by projectin the question in the memory space we get a dense vector representation of a certain size calling it $k$, then we need to generalize, so what part of that network respond to that part. How do we do? We take fact and transform them again into vectors the one above of size $k$. What we do is dot product them and we run softmax, then we go back to the fact, we reembed them into a different embedding, there are like the $V$ in self attention, we do convex combination multipling each $p_i$ with $c_i$ getting an $o$. 
At the end we decode the $o$, so obtaining a single words as answer.

![[Pasted image 20230414153226.png]]

## Extension to the memory networks

Three step reasoning mechanism, we do what we just did one time, the outcome is passed as intermidiate step, why we need to do that? Because we can't write in memory, so we need to implement like single steps.
![[Pasted image 20230414153946.png]]

## Memory nets for visual QA with attention

![[Pasted image 20230414154133.png]]
![[Pasted image 20230414154301.png]]

## Neural Turing Machines 

Input can be fact we want to memorize or queries, controlled is still RNN, this time we can leverage a memory a matrix in which we can read and write, into controler we have adaptive components heads, the heads has attention mechanism for writing and reading, so they can tell what is important to read and what is important to write. This is fully differentiable

![[Pasted image 20230414154408.png]]

### Neural controller

The states are saved into registers like memories, the thing up is the memory (memorizing vectors). We can change the direction of vector to show the change of informations. Trough execution we can see that memory changes, the key to differentiability is that whe we transform memory, that is the noticiable change.
![[Pasted image 20230414154709.png]]

### Memory read

Request that comes from the controller, how does a cpu ask to read memory? Want an address. To read we have to do convex combination, those are output of softmax, the memory read there is from the most *blue cells*, we can use a key query mechanism, creating a key for each element of the memory then we do key query matching (Context based addrexing). Neural Pointers Network introduce pointers to specific portions, still using a specific attention.
Still fully differentiable.
![[Pasted image 20230414154944.png]]

### Memory write

Given that we have some value that want to write, we have a vector of $\alpha$ attentional values, the job of controller is to write. But how do we decide where to write them, the indexing, how to address the memory?

Attention, but not only that, when we want to read we are doing is context based addressing, **we need to keep track of the position**, the spatial dimension, other than context based, idea is to understand how do we access memory and map to differentiable way.
![[Pasted image 20230414155317.png]]

## NTM Attention focusing

![[Pasted image 20230414155828.png]]

![[Pasted image 20230414160011.png]]

This is for location based indexing using shift distribution filter : eg [-1,0,1] shift our attention to the right. Sequential indexes. Then we sharpen, increasing difference between high number and low numbers
![[Pasted image 20230414160026.png]]

## Practical uses:

A nightmare to train.

Pondering networks : how to go from point A to point B. Pondering mechanism is telling to the network to take time, som proble require more other less, so the network iterate on itself depending on the complexity of the task with an inductive bias. 
![[Pasted image 20230414160342.png]]

### Take home lesson

Attention is really all you fucking need (?)

```ad-summary


```


---
# References

