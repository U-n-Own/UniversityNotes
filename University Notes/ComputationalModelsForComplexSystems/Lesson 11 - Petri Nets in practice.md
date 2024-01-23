Date: @today

Status: #notes

Tags: [[Dynamical and Complex Systems]]

# Flexible Manufacturing Systems

Different product are made by shared resources, we aim to coordinate these products and machine to behave in the most optimized way to produce more in short time. Without having deadlocks and some other bad stuff, cirucular situations are bad.

![[Pasted image 20230421145841.png]]

## Deadlocks prevention

![[Pasted image 20230421150047.png]]

In the paper is presented an algorithm that prevents deadlocks

Idea is to design the system with PN.

![[Pasted image 20230421150128.png]]

Constraints for these model: our Working Process has initial and final state, we can't have loops, one resource taken into the process at a time for each concurrent `state` so we can have two but one is in the robot arm scope and the other in the two machines scope..

We can do composition of two petri nets by adding share of resources, $r_2,r_3,r_4$ in his example, so by merging the places that share the same resources. Now these are actually shared in two WPs.

If a product lines is Deadlock free, since these go straight can't have selfloops and hence no deadlocks, but when we compone two the same resources can be used in "different" places.

![[Pasted image 20230421150749.png]]

Example of deadlock
![[Pasted image 20230421151121.png]]

Reaching this production stops, no legal actions avaiable. We would like to identify situations that leads future deadlocks. 
Introducing notion of *siphons*: 

- Pre : production positive
- Post: production negative

![[Pasted image 20230421151329.png]]

```ad-important
Theorem: A petri net is deadlock free iff for each rachable marking m and for every minimal siphon S hold: $m(S)\neq 0$
Note that this is not a general property.
```

![[Pasted image 20230421153249.png]]

Since they are waiting for places that are empty (siphon), we want to fill that places but the fact that we need to fill them needs to unlock those as we can see from the graph: output are blue and input request are pink (consumer). An empty sihpone creates deadlock.

Solution, create a place for the siphons.

![[Pasted image 20230421153623.png]]


![[Pasted image 20230421153734.png]]

Just adding more places slow down the process and doesn't guarantee the absence of deadlocks, an idea is to add these places one time and then use `lock`.

Main idea is to have a place invariant that make constant and not equal to zero the number of tokens inside siphons, identifiying dangerous sihpons, adding meta-control and doing this.
![[Pasted image 20230421154303.png]]

## Different analysis techniques over modeling languages

We've seen PN, Gillespies algorithm and equation system within multiset rewritig rules and model checking.

We can take this model run Gillespies algorithm to see the average production rate of a certain product. Or we can can check with models checking if properties holds and so on

>[!info]
> 






---
# References

