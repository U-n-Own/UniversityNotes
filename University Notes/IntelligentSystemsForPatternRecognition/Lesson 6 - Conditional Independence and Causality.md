Date: [[2023-03-03]]

Status: #notes

Tags: [[IntelligentSystemsForPatternRecognition]], [[Probability]],[[A.I. Master Degree @Unipi]]

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

Where $\bot$ means conditional independance, for far away variable there is no depedance, how do define far away : you're not a children. 
So apart parents others isn't needed: eg. children of children aren't needed

![[Pasted image 20230303152541.png]]

Example of students life : Decisional process in student life.

![[Pasted image 20230303152814.png]]

Why is that? Party and Study are marginally independant, if something happens to Headache they become dependant from each other, there are two causes for Headache, as soon you know the consequences (Headache) and you're not partying then you know immediately that you're studying, seeing a shared effect makes a connection between them! This is called **Marrying of the parents**. If we take tabs then we actually knows that headache is conditioned so you're also considering party and study. Like a chain of reasoning

## Another example of Conditionally Independance

![[Pasted image 20230712195407.png]]

## Markov Blanket

These can be used to say what are the variables that i should care if a consider a certain RV and what doesn't matter. Very large Bayesian network gets simplified a lot by these inferencial processes. 

We can call the other nodes that with $A$ have children Co-parents.
![[Pasted image 20230303153525.png]]
Now we can perform automatically factorization of Joint distributions
![[Pasted image 20230303154241.png]]
This makes us reason in a mechanic way.
We taken a joint distribution and simplified it in a product of simpler ones, because they entails less arguments

### Sampling a BN

I can sample values of all random variables, i pick a certain topological ordering, where we may take the ones without parents and we start "drawing" from distribution that gives us samples, starting from *party* 
because is uncoditional, we use ∼ to define the drawing from distribution

Generating i-th sample for each variable in graph above

1. $pa_{i}∼ P(PA)$
2. $s_{i}∼ P(S)$
3. $h_{i} ∼ P(H | S = s_i, PA = pa_i$)
4. $t_i ∼ P(T | H = h_i)$
5. $c_i ∼ P(C | H = h_i)$

This is **ancestral sampling**

### Fundamental BN structures

We are intrested in structure that give us some intresting information we determining conditional independence.

- ***Tail to tail*** (common cause): Our intresting factorization will be

$$P(Y_{1,}Y_{3}|Y_{2})P(Y_{2}) = P(Y_{1}| Y_{2})P(Y_{3}|Y_{2})P(Y_2)^{\color{red}1*}$$

Observation of $Y_2$ makes the other ones conditionall independent $Y_{1}\bot Y_{3}| Y_2$
When $Y_2$ is observed we say that **block the path** from $Y_{1}$ to $Y_3$

![[Pasted image 20230307112625.png]]

- ***Head to tail*** (Cause-effect): Actually the two structure are equilivalent. $Y_2$ is blocking the path from $Y1, Y_3$ in this way we can see it clearly! So these two structure ***Entails*** the same conditional indepedance relationship. So we must be very careful on how are the structures between nodes.

$$P(Y_1,Y_2,Y_{3}) = P(Y_{1})P(Y_{2}| Y_1)P(Y_{3}| Y_{2}) = \color{red}{1*}$$

![[Pasted image 20230307112654.png]]

- ***Head to head***
As soon you observe an effect you know something about the common causes, which make causes dependent to each other.
![[Pasted image 20230307112753.png]]
In this case if we observe $Y_2$ or any of it's *descendants*, because if you observe a descendance you're introducing it by marginalization and it will be observed and unlock then path is unlocked between $Y_{1,}Y_{3}$

### Derived relationship

Considering this
![[Pasted image 20230307114632.png]]

*Local Markov Relationship*                                                                        *Derived relationship*

- $Y_1 \bot Y_3 | Y_2$                                                                                              $Y_{1}\bot Y_{4}| Y_2$

### D-separation
![[Pasted image 20230307114920.png]]

Where $Z$ is a list of nodes, given a path is sufficient to find at least one node that blocks it.
*d-separation* holds if 

- the path $r$ contains **Head-to-tail** structure $Y_{i}\to Y_{c} \to Y_j$  if $Y_{c}\in Z$
- 
- the path $r$ contains **Tail to tail** structure $Y_{i} \leftarrow Y_{c}\to Y_j$. If $Y_c \in Z$

- the path $r$ contains **Head to head** stucture $Y_{i} \to Y_{c}\leftarrow Y_j$. If and neither $Y_c$ nor it's descendants are in $Z$

 Generally in BN, two nodes $Y_i,Y_j$ are d-separeted by $Z$ (a set of nodes) $\iff$ all undirected paths between $Y_i,Y_j$ are d-separated by $Z$: $D_{sep}(Y_i,Y_j | Z)$
 
But since set of nodes $Z$ could be anything we can further define something we already seen: Markov Blankets!

These are the minimal set of nodes which d-separates $Y$ from all the other nodes, making it conditionally independant of all other nodes in BN, remember that these comprehend the parents, child and coparents of our node.
 
![[Pasted image 20230307115425.png]]

So ***Markov Blanket*** shields all the paths with the minimum number of variables.

### Are directed model enough

They don't represent conditional dependance and the acyclicity constraint don't make us represent *symmteric dependancies*,

![[Pasted image 20230307120054.png]]

As soon we try to model the $Y_{2}\bot Y_{4}| Y_{1}Y_3$ we hit a cycle!  

Why?

## Markov Random Fields

What is the undirected equivalent of d-separation in directed models?

![[Pasted image 20230307120244.png]]

The node2s A,B are conditionally independant given C, so the markov blanket of a node is the list of neighbors. We need to find graph structures that makes $\color{pink}Cliques$ that are independant to each other, we want the maximal cliques.

![[Pasted image 20230307120626.png]]

### Maximal clique factorization

We want to factorize cliques now, so:

$$
P(X) = \frac{1}{Z}\prod_c \psi(X_c)
$$
Since we're seeking for local properties of some nodes in our graph, we can use potential function define them over some cliques, since potential function are not probabilities, other than the need to normalize they has to be defined properly to work well. Because they needs to weight which configuration of our local variable is preferred.

![[Pasted image 20230613193632.png]]

![[Pasted image 20230613193708.png]]

Boltzmann distribution is another example of widely used potential function.

### Directed to undirected

Getting rid of direction is easy, if you simply cancel the orientation, when you have a $V$ structure, you create a connection between parents (*Marrying the parents*) and reorient the arrow from parents to child node, to make sure they ends in the same clique, when we remove the direction we and add edges we **Moralize** the graph, this will be an equilvalent class for each directed type of graph, sometime changing orientation don't changes things.

![[Pasted image 20230307121143.png]]

## Learn causation from data with Bayesian Nets
![[Pasted image 20230307121916.png]]

### Structure learning problem

We're in the green top of the above image.
We want to learn some structure given the RV, all observable. We want to infer the arcs,  and then learning some parameters, now we're intrested in the structure.

#### Approaches structure finding

- Search and Score: Search in all the space (NP-Complete problem), we can call a model selection problem, we find for a particular structure that is simple enough, needs to score how complex the graph against how good is. So by doing some regularization on the graph we can say.

- Constraint based: Constraint the graph starting fully connected and start to deleting edges using test of conditional independance. We're taking test on conditional independance.

- Hybrid: The problem of search and score is complicated, we use constraint based to find an initial structure and then we use a score to measure

##### Scoring function
 
We want the scoring function to have **consistency** (Same score for graph in same equivalence class) and **Decomposability** (locally computed, so we want to use dynamic programming)

Approaches are:

- Information theoretic: based on likelyhood plus some model penalization term for complexity
- Bayesian: score using posterior

![[Pasted image 20230307123644.png]]

The last term is a penalization usually drawn from a Dirchilet distribution.

#### Search strategy

The AIMA book contains a lot of search strategies.

- *Constraint search strategy*: start from a structure and modify locally iteratively (because we keep the score that aren't modifed for dynamic programming reasons). Some search algorithm are simulated annealing, greedy hill-climbing and more

- *Constraint search space*: Know node order; as we seen we can search by parents. (In the markov blanket)

#### Contraint based 

Here we work on test of conditional independance $I(X_{i,}X_{j}| Z)$, from statistichs we need to define a good statistics for conditional independance and testing if there is an edge between two nodes.

![[Pasted image 20230307124221.png]]

This statistic will say how $X_{i,}X_j$ are influenced by $Z$, we want to determine **Mutual information** (see information theory)

#### Testing strategy

If the edge between two node and survive the test, we keep it there and test the other edges for uncoditional independance, if someone fails we cancel the edge, otherwise we will reach the edge we tested initially, and so with 2,3,4 variables. Super Exponential

- *Level wise testing*: with PC algorithm avoid super-exponential, going from order of increasing size.
- 
- *Node wise testing* : single edge at a time, until exhaust all

![[Pasted image 20230614172908.png]]


### Take home lesson

Directed graphical models represent asymmetric causal relationship between RV, is difficult to asses conditional independances in $V$ structures, undirected graphical models are easy to asses conditional independances

```ad-summary


```


---
# References

