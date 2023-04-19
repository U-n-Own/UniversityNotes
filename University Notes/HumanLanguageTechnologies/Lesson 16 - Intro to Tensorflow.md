Date: [[2023-04-18]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]]

# Structure of the library

We can run in parallel on different GPU.
![[Pasted image 20230418103744.png]]

## Some TF history

Draw a table of graph exectution against eager execution default in tf 2.0:

Graph Execution:
- Computation is defined as a static graph before running the program
- Operations are executed in a specific order determined by the graph
- The entire graph is compiled and optimized before executing any operations
- Designed for high-performance computing and large-scale distributed systems

Eager Execution:
- Computation is defined dynamically during runtime
- Operations are executed immediately as they are called
- No need to compile or optimize the entire graph beforehand
- Designed for easier debugging and prototyping

In summary, graph execution is more suitable for large-scale distributed systems where performance is crucial, while eager execution is more beginner-friendly and allows for easier debugging and prototyping. 

## Graph and computation

1. Assemble a graph, where nodes are function that our gradients flow and gets "updated"
2. Execute operations in the graph

Main benefits of graphs are to save computation running subgraphs, so breaking computation into small differentiable pieces.

TF execution has some objects: `Variables`: Parameters of the model ,`Model`: Our mathematical function,`Loss measure`: Guide for optimization of gradients,`Optimization method`: criteria to be used to make update, `Function`: decorator for python to use JIT compilation for running faster

## Tensors and data flows

Tensors are $n-dimensional$ arrays, we talked about it extensively in [[Lesson 15 - Intro to PyTorch#PyTorch]] 

```Python
import tensorflow as tf

def a():
   return tf.add(3, 5)

tf_a = tf.function(a)
tf_a.get_concrete_function().graph.as_graph_def()

node { name: "Add/x”
  …
 attr { key: "value”
        value { tensor {
  …
int_val: 3 } } }}

node { name: "Add/y” …}
```

This would be our graph of computation in that case, simple enough

![[Pasted image 20230419142835.png]]

## Autodiff

Usually we manually evaluate our derivative, but we could also apply the definition of derivative so having numerical differentiation or could use symbolic approach like matlab, that return another symbolic expression that will be our derivative

![[Pasted image 20230419144154.png]]

These symbolic expression can be represented with graphs, so we could use graphs to do it automatically.
Again we saw autodiff in the other course [[Lesson 15 - Intro to PyTorch#Automatica Differentiation]] and [[Lesson 15 - Intro to PyTorch#Building Automatic diff graph]].

### Reverse accumulation in autodiff

![[Pasted image 20230419145956.png]]


### Autographs

Technology behind creation of graphs in tf, in there we have `tf.function` putting a decorator when annotating a function with tf.function


>[!info]
> 






---
# References
