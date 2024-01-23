[[Fine Tuning]], [[LLMs]], [[Efficiency]]
## LoRA

LoRA method that provide a modular approach to fine tuning and provides capability of doing easier [[Transfer Learning]].

### The math catch

So initial pre-trained weights and the fine tuned often exhibits **Low intristink rank** so this means we can approximate this matrix with a low ranked one, and if a matrix has low rank, so by having fewer independent columns we have the property to decompose it in two smaller matrixes, this lead to the idea that the "$\Delta$" between the two matrix before FT and after can be represented as the matrix product of two different matrixes much smaller than that, so by updating these two as neural networks we improve computational efficiency 

### Dimension rank decomposition

![[Pasted image 20231127165819.png]]

What we do is to apply these LoRA adapters to existing feed forward networks, freezing the original FFN and using these network for training

![[Pasted image 20231127165958.png]]

So by adding a layer of trainable parameters while maitaining original in a frozen state, so the $W_A$ and $W_B$, utilize low-rank weights vectors.

### Another illustration from the paper

The input vector $d$, is processed trough both pretrained and LoRA fine-tuned, so we backprop trough the small things and we profit a lot.

![[Pasted image 20231127171138.png]]

## Hyperparameters : LoRA $\alpha$ & Dropout

So now what are the parameters that regulate our two smaller matrixes? The rank $r$ of course and a parameter $\alpha$ that scale the matrix

Usually for $\alpha$ is advice to use $16$ as the rank, also sometimes is advice to have a ration of $r$ and scaling that is around 1.

Higher rank counteract efficiency but performance increase when increase ranking...

## Q-LoRA

This extends it by adding *Quantization* on the weights values of the original network, the higher resolution used is Float32, and lowest resolution is Int4 bits, so if we have weights that are much less demanding in precision we could reduce both memory usage and speed up the learning but at the cost of precision

#### 4-bits NF4 Quantization

This type of data can be used to store weights in 3 steps

1. **Normalization and Quantization**: Adjusts weight to zero mean and unitary variance, since 4 bit can only store 16 numbers, we map our weight to to this 4 bits storage zero centered and distributed but instead of storing the weights we store the nearest position

example 

![[Pasted image 20231127172418.png]]

0.2121 is closest to 0.1997, which is the 10th position. Instead of saving the FP32 of 0.2121, we store 10.

Obviously there is a loss in the data when we do quantization step which is high-res to low-res compression to avoid too much change in the weights distribution we quantize weights independently in small blocks

2. **Dequantization**: The 4-bits is applied to the frozen weights of the model and the adapter will stay trained in FP32, after the training is done the weights are dequantized

![[Pasted image 20231127172810.png]]

3. **Double Quantization**: Yeah you just got it, quantize the *quantization constant* we got in the prev quantization step, since we do this in block : 64 params/weights per block... 

```ad-check

Sanity check calculations

> Here is the calculation: We have 64 weights in 256 blocks which are 32 bits which is 32/(64*256) which is 0.001953125
> 
> We have 8bits for 64 weights which is 8/64 0.125
> 
> If we add it up 0.125+0.001953125 which is 0.127 approximately

```

4. **Unified memory paging**: At this point we could use nVidia unified memory feature, allowing offloading on CPUs (GPU -> CPU) page transfer when GPU runs out of memory so managing spike of memory usage in GPUs they are very costful, well i could say this is a very nice stuff but my CPU offloading is not the best since FUCKING PEOPLE LOAD 100% CPU AND I CAN'T FUCKING WORK I HATE STUDENT CLUSTER PLEASE HELP