Date: [[2023-03-08]]

Status: #notes

Tags: [[Parallel Computing]],[[A.I. Master Degree @Unipi]]


# Vectorization techniques

Vectorization can have large advatages in some kind of codes. Every times you have a very large collection of item in vectors and have to perform computation we want to compute fast.

Suppose two vectors $V(n), W(n)$ and we want $K(i) = V(i) + W(i)$, if we execute in a loop this code in a standard processor, we need to execute the loop into these mathematical operation, these happens in registers, that are limited (10,32,128) but not large numbers. We need to load the first $v_i$ into a register then load the others, add them, store into $k_i$, do some check for the loop. We get 7 instruction per iterations. This happens in common processors.

I can load a vector register that is basically  like this: |__64__|__64__|__64__|__64__|

So we take the numbers into the registers and threat them like vectors and with 4 ALU, we can compute in 1 pass 4 operations. In common processors we have a 4x64, 8x32, 16x16, 32x8, as vector dimension.

What's the point of using ***vector units***?
Well in 3 cycle we do what we did in 3x7 cycles before! So vector processors work in parallel like this.

We need code as well to make this work, to tell to use vector instructions, also if we have something like $r[i] = r[i-1] + something$, this can't be executed in parallel.

### What can't be vectorized?

- No dependancies
- No breaks inside the loop
- No conditional statements 
- No call to libraries
- No overlapping pointers

If these are satisfied, then we can tell g++ -O3, the flag `-O3` that adds a lot of optimization including unrolling of loops

## Vectoring

```ad-todo
Rewatch first hour of lesson!
```


## Type of Computations


We can work with some pattern to work on them that's the *high level*


```ad-todo
Code assignment:
Divide and conquer but each time you divide you use threads to compute
```

# Divide and conquer approach

Take a vector, we want to compute map of this vector using threads.

For a length $n \gt limit$ or $n \leq limit$ we have:

```C++

for(i = 0; i < n; i++){
	v[i] = f(v[i])
}

```

Of course we use limit for finding the base case. When taking subvector from our vectors we declare a pointer to the subvector (0 to mid) (mid+1 to n-1). So the conquer step works on the pointers, we keep these.

Let have a binary three, we have the first that goes to the .get call and th$e two active one, for a total of 3, then the two do the recursive producing 4 and so on, if we have a height $h$, on our binary three we have a frontier of $2^l - 1$ nodes that are the current working, but if we go ahead we will have the $2^{l-1} - 1$ blocked (waiting)

#### Overheads and problem in this?

- Joins and barriers are more expensive than asyncs.
- When doing recursive call there is the function on the stack and all the stuff

## Task Pool

No primitive implementation in C++

With task pool we can divide the task like we saw in the struct

```c++
struct{
double *v;
int start;
int stop;
}
```

### Malloc in thread

JMalloc library to work with dynamic allocation in multithreading

### Move semantics

There are some data structures, that do not copy in memory but assign new name to the data structure.

### Take home lesson

```ad-summary


```

# Data Parallel

Last time we talked we didn't discussed about the time of those 

See Computation.xopp 
![[Pasted image 20230314154644.png]]

- Map : $L = t_{split}+ \frac{k\;t_{subproblem}}{num}+ t_{merge}$

**Stencil** :We want to compute a point and we consider all the point in the neighbor

### Google Map|Reduce

There is a map function $f$ and $\oplus$, where $f(x) \mapsto <k,v>$, and the reduce take $<k,v_{1}> \oplus <k, v_{2}>$, so if you've a 
$$\{X_{i\}}\mapsto \{<k',v'> ;<k', v''>...<v>\}$$


For example

$$
f(x) = 
\begin{cases} 
<even, 1> \text{if x is even} \\ \\
<0, 1> \text{otherwise}
\end{cases}
$$
$$
\oplus = +
$$


We do a $map(reduce(f))$ 

So we map the vector, we know how to parallelize, but what about the reduce step?
We can apply the addition in a three like way on the vector and get the <k,v>. But there is a step in the middle, after we computed the map, we can sum only the one with the same key! We must order them, can we do in another way?

We compute the $hash(k)$ and send it for later reducing. For example if in a subarray i get twice the same $<k_1,x+y>$

In general 

$$
reduce \; \forall <k_{i, x>} \mapsto \; <k, \sum x>
$$

Let's assume to have more "reducers", that apply a $\oplus$ operation on their following "key".
So we're computing a set of reduce: $map(...) \text{ over all the keys} \; k$, taking the map over all the keys i've to then *parallelize* this. There are many ways.

### Parellelizing google Map-Reduce

Lets assume a database of keys  : $D_{b}= \{(a,1), (b,1), (a,1), (c,1),...\}$ we can organize a three of reducers, and work on those subtrees with the reducers.

![[Pasted image 20230315113915.png]]


Another method is to use an hash function, for example $a \mapsto 1_{reducer}$, $b \mapsto 2_{reducer}$, $c \mapsto 1_{reducer}$

![[Pasted image 20230315113445.png]]

Another solution can be


A parellel program is basically a conjuction of **Patterns** : maps, reduce and more that we saw and will see

Eg:
![[Pasted image 20230315114629.png]]

Transforming the code from composition of patterns is fundamental in parallel programming.
The main take away message on google maps reduce, is to: watch your problem and solve it with building blocks you know, then try to optimize these things

## Flume paper

In the paper called Flume, we can se a technique called ***map-fusion*** 
![[Pasted image 20230315115315.png]]

$f,g$ depends on the users. Let's assume i spend $t_{f},t_{g}$  as the time i spend to apply the maps.
![[Pasted image 20230315115422.png]]


![[Pasted image 20230315115606.png]]

The column splitting produces overhead, what if we do all in one go?
![[Pasted image 20230315115718.png]]

### Map fusion theorem

Every time you have to compute some **stateless** function one can group all the function in an unique map, this reduce greatly the overhead of loading data in our computation.

![[Pasted image 20230315121455.png]]


If i have a pipeline with more stages and i want to optimize it i can do something like this:
![[Pasted image 20230315121617.png]]

Our example pipe:
![[Pasted image 20230315121536.png]]

#### Properties of computation

- Functional properties: Determines *What we compute*, if we consider a $map(f)$ followed by $map(g)$ what we compute is $g(f(x))$ $\forall x \in Data$ 

- Non-Functional properties: *How we compute*, if we decide to do it in parallel: $nw_{f}, nm_{g}$.
 
Speaking about non-functional properties. Each pattern has some peculiarities, and wekenesses (overheads). For the map we had overheads of forks and joins, also the amount of overheads depends on the OS and the hardware. 

Speaking about overheads we must consider: load balancing

### Load Balancing

What is the problem? I've my array with a number of threads to compute a map on it, so each thread has to compute a chunk
![[Pasted image 20230315122306.png]]
So basically we have some threads that finish or do least computation regardless of the others, but i want that each of them take about the same amount of time, so we have to balance the computation between the threads: eg multiplication by zero is really fast. So we're maximizing efficiency in this case.


eg. Colors on the mandelbrot set 
![[Pasted image 20230315123021.png]]

To color the set is given by coordinates ($x,y$), i can assign a thread for each subset of our image, those that compute the colors inside work much more than the one out of the set!
![[Pasted image 20230315123140.png]]

Typical umbalancing! Well a solution can be to do a round work on the entier picture so each thread takes every time a different part of the calculation.
![[Pasted image 20230315123309.png]]

This is much better than assigning chunks of an array to the different thread, because we're kinda making them work equally.

### Static strategies

![[Pasted image 20230315123735.png]]

- Block
![[Pasted image 20230315123944.png]]

- Cyclic: round robin 


### Auto-Scheduling

Suppose in the queue there are task with similar amount of time to be completed, with two workers
![[Pasted image 20230315124338.png]]

With the round robin would be worse! With the one above there is a little of unbalance (6t against 5t).
![[Pasted image 20230315124526.png]]

So each one search for something to compute for itself, with autoscheduling we can be more efficient with the other is fast but not efficient, can we combine them?

Yes!
![[Pasted image 20230315125024.png]]

This techniques are dynamic techs we're out of the static world.

### Job stealing

A last technique that combines static with dynamic is **Job stealing**
Let's split an array in $k$ pieces and give them to some threads

![[Pasted image 20230315125318.png]]
This is what happen in static strategies, but now when there is some thread that has no work he can steal some task from those that are still computing.


---
# References

