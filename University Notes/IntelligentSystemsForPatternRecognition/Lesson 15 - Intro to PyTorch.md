Date: @today

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]][[Programming]]

# PyTorch

Ecosystem for ML: Tensors manipulation the main data structures used in DL, GPU support, Autodiff, high level API.

Tensors generalize multidimensional array equivalent to `np.ndarray`, we can give type to these tensors during init. We can feed them to cuda if have GPU or to CPU.

### Operations

Tensor operations: $+,-, * \;\text{element wise},\; @ \;\text{matrix mul}$

### Indexing

Slicing, indexing , placing conditions, we can even slice specific portion of the tensor by doing something like this

```python
a[:t_max, b:b+k, :]
```


### GPUs

Always select a single GPU if others are using it you could broke someone's else experiment

Manual selection: ('cuda:0', 'cuda:1'...)

Changing the shell env `CUDA_VISIBLE_DEVICES=0`

Or automatically choose the GPU that fits better for our model or experiment. Best of the ones.

### Automatica Differentiation

`torch.autograd` knowing a bit about autograd is very useful to investigate what's happening, every computation is saved into a graph  (tree), function nodes (Operation on tensors, that are our input to this graph) and we can call a backward method in each node in the network that gives us the backpath in the graph from that specific node.

Node are made this way: `autograd.Variable`: (data, grad, grad_fn)

Each function has a forward application and backward application, we can get the grad to explore what's happening to gradiet in that point, basically debugging.

We can set a parameter `requires_grad` to specify if the gradient should stop at that point, so freezing the parameters at that point during training. And only the parameters that have the requirements will be used to continue training.

- In-place modification break the autodiff!

### Building Automatic diff graph

We can instatiate a Variable (Tensor), a new node is automatically added to the computational graph.

-- image of graph and code

## Torch.nn

This module `torch.nn` contains layer of network, precoded perceptron, LSTM and much more, loss function: cross-entropy, regularization techniques and optimizers, even second order optimizers (LFDS).

### nn.Module

Base class for all neural nets, a module contains parameters, and one cna compute the output by using nets like functions `y_pred = net(X)` because of the `__apply__` method is overridden computation is performed by forward, but we can also call directly forward manually this is more useful for debugging by using hooks, detecting vanishing or exploding gradient and interrupt training or do fancy stuff.

How to define hooks?

`register_forward_hook()`
`register_backward_hook()`


## Static vs Dynamic graph

When using static: faster, the debug is gonna get way harder, code is compiled so can't print value of tensors, but dynamic graph is good for debugging but slow.

### Pytorch lightning

A library that needs only optimizers and loss function model and data and will perform training in an automated way, have cool features to use all those can handle.


### Take home lesson

```ad-summary


```


---
# References

