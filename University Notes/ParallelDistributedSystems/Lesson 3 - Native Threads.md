Date: [[2023-03-01]]

Status: #notes

Tags: [[Parallel Computing]]

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

``` ad-important 
Assignment:

Suppose to have a vector like this

Vector<int> v(n); where n is number of items

We want to get a w(n) that has a transformation such that w(n) = f(v(n))

Take a number of threads from command line, and make such that if you get n threads each of them compute one of the n items, v and w are shared between threads.

tip. for(i=start[index]; i<stop[index]; i++){...}

Use chrono to measure time and compare against the sequential


```







### Take home lesson

```ad-summary


```


---
# References

