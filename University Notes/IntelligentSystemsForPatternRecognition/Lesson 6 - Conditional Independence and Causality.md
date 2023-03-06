Date: [[2023-03-03]]

Status: #notes

Tags: [[IntelligentSystemsForPatternRecognition]], [[Probability]]

# Bayesian network and Markov fields

Why are we intrested in these network? Because we can reason about cause effect or constrained problems

![[Pasted image 20230303150022.png]]

Two set of nodes, empty and shades, observed RV and unobserved,hidden ones.
Given a joint probability distribution of the variables connected provide a possible factorization of the joint distribution, these can be factorized in a lot of ways, not all factorization are *consistent*.

What we can see is that $Y_3$ has some dependences with his ancestors. So we are surely intrested in $P(Y_{3}| Y_{1,}Y_2)$, those of $Y_1,Y_2$ are not dependant from anyone, we're just storing the edges and the empy nodes. Simply factorize on the dependances.

In this case if all $N$ are independant then we have $N$ variables each with $k-1$ possible values
![[Pasted image 20230303150908.png]]

Sometimes we encounter a model called **Naive Bayes**, when we get the class the attributes become all independent!
![[Pasted image 20230303151042.png]]

A *plate* is basically a replication of bayes classifier, this tends to become very complicated, by looking at plate diagram we can then describe what we got.
![[Pasted image 20230303151227.png]]

We have a generation by $\mu$ and $\sigma$ and $\pi$ tells us which sample, this provied a picture between RV and parametrization.

## Local Markov property

Let's reason on the structure of the Bayesian networks
![[Pasted image 20230303152232.png]]

Where bottom means conditional independance for far away variable there is no depedance, how do define far away : you're not a children. 
So apart parents others isn't needed: eg. children of children aren't needed
![[Pasted image 20230303152541.png]]

Example of students life : Decisional process in student life.
![[Pasted image 20230303152814.png]]
Why is that? Party and Study are marginally independant, if something happens to Headache they become dependant from each other, there are two causes for Headache, as soon you know the consequences (Headache) and you're not partying then you know immediately that you're studying, seeing a shared effect makes a connection between them! This is called **Marrying of the parents**. If we take tabs then we actually knows that headache is conditioned so you're also considering party and study. Like a chain of reasoning

## Markov Blanket

These can be used to say what are the variables that i should care if a consider a certain RV and what doesn't matter. Very large Bayesian network gets simplified a lot by these inferencial processes. 

We can call the other nodes that with $A$ have children Co-parents.
![[Pasted image 20230303153525.png]]
Now we can perform automatically factorization of Joint distributions
![[Pasted image 20230303154241.png]]
This makes us reason in a mechanic way



### Take home lesson

```ad-summary


```


---
# References

