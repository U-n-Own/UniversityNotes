Date: @today

Status: #notes

Tags: [[Parallel Computing]], [[A.I. Master Degree @Unipi]]


## Computer Scientists are Sys. Programmers

In a more computer science based context we're those who produce the libraries that will be used in production! 
![[Pasted image 20230322112039.png]]

## Using futures in task pools

This pattern combines futures, that return the computation when asked using `get_future`. Combining packaged tasks 
![[Pasted image 20230322112716.png]]

example
![[Pasted image 20230322112948.png]]

## Termination problem

In posix there is a method called cancel, that can be called if ones have the handle for the thread and can kill the thread but, usually threads cannot be killed.
What can we do? 

Problem: I've a queue of tasks and then i've threads executing in while loop, trying to get the task, compute it and back to work on another task. How can we end these? 

We could use a shared variable boolean to make the system know if we want to keep going. Or we can use a conditional variable and put a wait on a certain point, so wait if false, ok if true...
When a notify all we reawake all the threads they'll pass the wait, so if there is nothing else to compute we can put the flag to true, on the condition variable we do notify all that notifes all threads that want to pop from the queue and if q.empty is false we break, instead if there are elements we continue to work on that queue.

![[Pasted image 20230322113449.png]]

This is a notable problem because all terminate togheter.

We can add a certain mark at the end of the steam (EndOfStream). We make the thread exit. So when getting EDS in input we have to stop, we need to put as many EDS mark as many threads we have because we need to make sure that all threads are terminated.
![[Pasted image 20230322114129.png]]

What if i have to terminate this:
![[Pasted image 20230322114500.png]]

We sorta wait for all the traverse and wait if there is something to do, if there isn't then EOS are slowing because we're busy waiting
![[Pasted image 20230322114734.png]]

## A solution: Optionals

By doing this i can assign to $i$ an integer, with has_value we can get a boolean values to see if $i$ actually contains something.
![[Pasted image 20230322114942.png]]
![[Pasted image 20230322115139.png]]

## Another idea of using futures

This idea of futures use isnt that good,
![[Pasted image 20230322122155.png]]
This is not a good idea to represent dependancies, a better idea is to use

## Macro Data Flow Graphs
![[Pasted image 20230322122451.png]]

If i have a function $f$,  then i do $<z,k> = f(x,y)$.

![[Pasted image 20230322122642.png]]

This is for farm but what for maps?
In this case we have as **firable** $f$ at the start then $g_1,g_2,g_3$ and $h$ at the end.
![[Pasted image 20230322122815.png]]

We can imagine a unit that takes firable units send to task pool that return results
The duty of matching unit is only to give tokens to the $h$ until he gets 3 tokesn and then it can be finally fired, but has to wait all the 3 $g$, the graph of dependancies we've to see, can give us information about how much is the parallel degree of our computation.
![[Pasted image 20230322122957.png]]


## Structured Parallel Programming

This research line started in the 90s a community searching high performance community, then emerged also from a software community

- Some history: 
![[Pasted image 20230322123447.png]]


Some computation which we don't know a priori the computation time, we could take the graph of activities and transform it into the MDF Graph and exploit the structure there if we could parallelize better. 
![[Pasted image 20230322124011.png]]


MDF comes from Pisa here a lot of research is being done


## Magma Libraries

Gives MDF or Template implementation of map, and depends on dimension of the computation, these two implementation looks at the dimensions and can take the best for the dimension. These have different methods a lot of problem can be handled differently in these two way, like the *termination*, *load balancing*.
![[Pasted image 20230322124538.png]]




****
### Take home lesson

```ad-summary


```


---
# References

