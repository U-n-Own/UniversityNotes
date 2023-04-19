Date: [[2023-03-01]]

Status: #notes

Tags: [[Parallel Computing]], [[A.I. Master Degree @Unipi]]

# Threads as Objects

Today we get a more practical lesson,  threads in C++ are Objects, threads are about sharing and we want to protect the sharing, some mechanism we can use are syncronization, so 2 Threads can't execute the same part of code.

## How to connects ssh

Next week we will bring some code.
We can access the remote machine here: spm.di.unipi.it

We can access with ssh. Good thing to do is this:

```bash
	ssh spm.di.unipi.it 
	#Use this command then
	ssh-copy-id spm.di.unipi.it
	#Will be generated .ssh folder with id_rsa and id_rsa.pub
	# This wont require to use password everytime
```

## Threads (C++)

Use [C++ Reference](https://en.cppreference.com/w/) 

```c++
#include<thread>

new thread(f, arg1, arg2, ...)
f(args*[]){
.
.
.
return ...;
}

```

This returns a thread pointer thread **thread \*tid**, after a thread function ends the threads go in a *joinable* state the thread will block after the return.

Calling a thread and joining it is a time that is to be accounted, the time is on the tenth of microseconds : $\mu_{sec}$ 

## Sharing

If i have a variable $x$ declared globally any thread can access it.

To protect a piece of code when sharing, like update a variable we use *mutex* tools like semaphores


### Atomic Variables

```c++
include<atomic>

std::atomic_int acnt;
int cnt;
```

You can't do operations like $x = x + n$ on atomics integers but can do with $x =+n$

## Assignment

``` ad-important 
Assignment:

Suppose to have a vector like this

Vector<int> v(n); where n is number of items

We want to get a w(n) that has a transformation such that w(n) = f(v(n))

Take a number of threads from command line, and make such that if you get n threads each of them compute one of the n items, v and w are shared between threads.

tip. for(i=start[index]; i<stop[index]; i++){...}

Use chrono to measure time and compare against the sequential


```

# Forking threads costs

When forking a certain number of threads and joining them at the ends there is some time to take into account, also what is the time of computation? 
There is a tradeoff of setting
up the parallel work and the actual computation.
How can we compute the overhead of a program with some threads.
Program showed a lesson estimated $35 \mu s$

## Differential equation and threads

A common way to evaluate differential equation. eg Imagine to have a metal bar with a block of ice at one edge, and want to know the temperature at the center, you need to take into account that each cell of the bar depends on the cell near it and compute it with some cycle, and one cannot split the vector between different threads, in this case to evaluate some value of the temperature need to await that the other are computed.

eg : $$
v{[i]} = \frac{v[i-1] + v[i] - v[i + 1]}{3}
$$

## Producer-Consumer

Let's say there is a queue, protected, if consumer comes and see an empty queue, can wait, can enter in a loop, *active wait*. How do we solve problem? ***Barriers*** and ***conditional variables***

#### Barriers

From C++20, so when compiling you need "-std=c++20" flag.

Barrier is a way to await each other, executing a single portion of code when everyone has reached the barrier, a barrier has a number of participants (threads) and takes a second parameters that when run ends the barrier.

#### Condition variable

So we have a mutex variable $x$, we can *notify all* or *notify one*, if we're in the wait we must test for the condition. There can be some events that make such that when doing notify one of the threads go and the other go as well, without condition variables there is not test. 
See code at lesson for more insights.

## Asyncs

When we have a thread main and another thread they interleave and the scheduler works on the two in a simil parallel way. An [Async](https://en.cppreference.com/w/cpp/thread/async) is like a thread but  `async` runs the function `f` asynchronously (potentially in a separate thread which might be a part of a thread pool) and returns a [std::future](https://en.cppreference.com/w/cpp/thread/future "cpp/thread/future") that will eventually hold the result of that function call.

### Futures

A Future can be of a given type and is a "promise" that you get something of that type when async ends or when is needed we can get the values from the future that is subject under reification (from $future<int>$ ) is being returned the int.

```ad-seealso
The function template `async` runs the function `f` asynchronously (potentially in a separate thread which might be a part of a thread pool) and returns a [std::future](https://en.cppreference.com/w/cpp/thread/future "cpp/thread/future") that will eventually hold the result of that function call.
```

Remember lazy and eager evaluation, in lazy the values are returned only where asked, in eager all the results is actually computed, eg: Haskell has lazy evaluation with infinite lists. Futures are something similar but still eager.

If we have a part of code that is our **`task`** we can assign this as an op that is computed and then it's result returned as future.

### Transform

In the library algorithms we can find an algorithm called `transform` what this does (on can do) is to take a vector and trasform it into another with a given function. That's what we did in [[Lesson 3 - Native Threads#Assignment]] above. As well trasform can be used on all iterator (collections).

### Issues

When using an automatic variables assigned to a thread creation.

```c++
vector<int> v;
auto tid = new Thread(t,v);
```

## News for interpreters

For python we have jupyter notebooks that can be executed in blocks to better debug and code execution, this is a very useful thing, these are like document that can be executed step by step.
There is  ```Xeus CLing```


### Take home lesson

```ad-summary


```


---
# References

