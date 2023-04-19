Date: [[2023-03-21]]

Status: #notes

Tags: [[Parallel Computing]]


# Communication time spent matters

Time spent to communicate isn't negligible, this is another overhead we have to take into account, communication costs!

![[Pasted image 20230321142853.png]]

Usually we start a communication we setup the system and spend some time, if we take into account ethernet technology (100Mb solution). Once i setup the TCP connection i can transmit data at that rate, but we have to pay some millisecond of latency to open connection, plus the lenght of the data in the transfer rate, shared memory communications, could happen by someone that see a message in the machine and these are faster.

## Communication Hiding

Let's imagine to have some communication happening in parallel.
If you compute map of a vector and then have to communicate it you've to see the time
![[Pasted image 20230321144910.png]]

Pipilines patterns
![[Pasted image 20230321145412.png]]

What usually do CUDA from NVIDIA are opening streams
![[Pasted image 20230321145601.png]]

## Data Locality

- Spatial Locality:
- Temporal Locality: at certain time $t$ you can get back the usage of $x$.

![[Pasted image 20230321151936.png]]

Imagine to have some locality.
Simple strategies and policies that keeps data locality could help a lot, meaning that a lot of overhead is spared due to the concurrent activities.
![[Pasted image 20230321152958.png]]

![[Pasted image 20230321153848.png]]

## Thread Pools

Thread Pools are not native in C++, is something where we compute some tasks.

So we have a number of threads that execute a loop, and do a task
![[Pasted image 20230321154626.png]]

### Implementing a Queue

If we push two times into the queue we can't be sure that the contents are protected, the simplest is to use mutexes. In particular to instatiate the `lock-guard` and `unique-lock` classes to grant mutual exclusion.

Condition variables can be used to manage the syncronization.

### Impementing Tasks

A task is more than simple the code is $x$ data, $f$ the code and some results: $y$.

Common and best practices:
- In Std lib we can use:

```C++
auto t_1 = std::bind(f, x)
auto t_2 = std::bind(g, y)

push_back(t_1)
...
##########

\\ Example

auto f = [&](int i: float x){
	v[i] = fun(x);
} \\ by side effect

```

But instead of just using binds we can use `packaged_tasks`, this takes a bind and a future that can be used for later.

So we get the $pt$ then we do $q.push\_back(pt)$ and when we call we can get from the future

```C++
pt = packaged_task()
pt.get_future()

\\ When we need

(pt.get_future()).get()

```


### Take home lesson

```ad-summary


```


---
# References

