Date: [[2023-04-16]]

Status: #notes

Tags: [[Complex Systems]]

# Petri nets models concurrency

Petri's nets are pretty useful to model concurrent system, biological system but also manufacturing, each place we have multiple concurrent unit executing work in parallel. Parallelism and concurrency are different things but these components have to basically wait other to complete and pick resources and so on, for example ignegneering system have some controls, and we want to reason on what happens if some parallel components execute first. This motivate a lot of things
![[Pasted image 20230416155748.png]]

Another example is concurrent access to databases...


## Petri nets : a graphical language

Places are *type* of resources, and trasition are *rules* that makes transformation happen, so moving token from a place to another. This network we can see is an example. Token **lives** in places.

Resources can be of any type: state of something

![[Pasted image 20230416155831.png]]


## Transitions

Arcs from place to transition consume tokens and exiting arcs produce resources. A transition is basically chemical reaction or *multiset rewriting rule*.
![[Pasted image 20230416160039.png]]

## Firing a transition

Basically transition being happened

![[Pasted image 20230416160319.png]]

### Producer consumer and bounded buffer

Buffer stores products, producer produces them, and transition are performer autonoumsly moving the token in idle to producing and then producing stuff, the 5 tokens are the resources, everytime we produce we consume a "free" place, when all of the tokens are consumed the producer is stopped and consumer starts. Taking a product and adding a free token.

![[Pasted image 20230416160342.png]]


Another example are two machine that can read togheter, read while writing, but cannot write togheter, so we have [idle,read,write], now processes start taking action, also the sequence of transition is non-deterministic. So we have to cosider all the possible transition.

A transition for example would be : `write1 -> idle2`, but we don't want `write1 -> write2` this is *illegal* in our system. We can add a lock to ensure mutual exclusions, for both processes, in multiset rewriting is a new symble `M`, when $1$ enters and the other has the lock resource, then $1$ cannot apply rule to write...

![[Pasted image 20230416160919.png]]


### Spawning processes

This time we're spawning processes, starting with zero processes. $t_1$ represent the start of a new instance of process, (no ingoing arc so always enabled) spawn a process, $P$.

$t_2,t_3$ are transition that make process enter or exit the critical section.

- $p_1$ : processes 
- $p_2$: free mutex
- $p_3$: critical section

![[Pasted image 20230416161035.png]]


## Formal definition of petri nets

![[Pasted image 20230416161522.png]]

Petri nets are graphs, the $I$ and $O$ are basically mapping of items that tells our how many token we transform. Arc that we do not see are those that makes or consume zero tokens.

![[Pasted image 20230416161755.png]]

## Marking

We assume an ordering in the places of this vector, eg $p_1,p_2,p_3...$
![[Pasted image 20230416161824.png]]

Basically  a marking tells out the state of a petri net

![[Pasted image 20230416161928.png]]
### Firing a transition : more in details

Creating a new marking $m'$ and for each place we subtract the input tokens and add the output token to the old marking $m$. To represent all the possible steps from a marking to another we use the Post set, so a collection of possible markings...

![[Pasted image 20230416162003.png]]

### Example of post set

![[Pasted image 20230416162308.png]]


## Initial marking and reachable markigns

So like the possible initial states or reachable state in transition systems.

![[Pasted image 20230416162346.png]]

## Ordering

Marking can be vectors with natural numbers so we have an ordering between markings, vector of the same lenghts, we can say that a marking is smaller than another if for any place in the marking the frist is smaller then the second. We cannot compare if this doesn't happen on each position. 
![[Pasted image 20230416162610.png]]

## Questions on PN


Particularly useful if we want to infer information about states space, for example these questions like *boundedness*: is there a finite number of possible states?
Or *place boundendness* are some of resources of a given type that we could create is limited?
Or there are transition that can fire on a certain reachable state (marking): Basically we want never that a system reach a state in which will be blocked because no rules can be applied.

And **Coverability** we will discuss later.
![[Pasted image 20230416162858.png]]

### Reachability

How can we check if something is reachable? We construct all the possible transitions and check if a state is reachable, this will be called *Rechability graph* of our petri net. So the graph obtained by apply the *Post function*.

![[Pasted image 20230416163713.png]]

Seeing the rechability graph we can indeed see the mutal exclusion is enforced.

![[Pasted image 20230416163844.png]]

There is a problem something with unbounded tokens: the reachability graph becomes infinite! And this would be a problem if this is the only way we can check things.

## Infinite but decidable!

Is being proved that we can check if a marking is reachable! There are algorithms that check reachability but are of exponential complexity, and we need to introduce efficient approaches with some approximations. 

Decidability is obtained because of the structure of our graph: *each node has the same number of neighbours.*

![[Pasted image 20230416164336.png]]


Every transition has it's arc, wecan represent transition like vectors, like points in cartesian plane. We can see how they are transformed the $t_1,t_2,t_3$.

![[Pasted image 20230416165956.png]]

And now we can do it for different marking by starting in each point in the cartesian plane, where the initial values of $<X,Y>$ are naturals. Given the regularity we can exploit this. We could add constrains, and this method within regularity converges, maybe we need steps exponential in number of transition but will converge eventually.

![[Pasted image 20230416170251.png]]

### Decidable is fine but exponential is not efficient

***Solution 1***: Some proposals to go fast are **overapproximations** we don't compute exactly possible reachable state but we comput a set slightly bigger including not reachable.
Given this if we don't find the set of intrest there will be not reachable, otherwise we can't say if the state is reachable or our state of intrest is in the overset that isn't reachable...

***Solution 2***: Another class of proposals are in the change of point of view, isntead to check if marking are reachable, we want to ask : can we reach a marking which is greater or equal to our marking of interest. This if verifiable in polynomial time but is weaker than *reachability*.


![[Pasted image 20230416170551.png]]

## Place invariants

This isn't intresting only to prove reachability but intresting per se: We want to try a group of places (a clique we can say) that the summed number of token the number will be constant. This is an invariant property. 

Something when from a token and a transition we generate more than 1 token we can say that the token before has a bigger *weight*, so we can assign weightning to our token if they are in some specific place.

This make the PN place invariant again 
![[Pasted image 20230416171626.png]]

![[Pasted image 20230416171131.png]]

### Formally: place invariance

A vector of weights, natural numbers $i$ such that overall weight of those in the initial marking wil be equal to the overall weight of the net in another marking of the same. This can be seen as the ***mass conservation*** property we know from chemistry. Our transition must preserve the mass into the net.

![[Pasted image 20230416171655.png]]

We can specify invariant different of them for different place in our petri nets (subsets). For example $p_1,p_3$ are a subset and the place invariant is $1$. Also we can speak about **maximal place invariance** if we're speakign about the entire net.

If we are able to specify place invariance for a specific place we can analyze how that place behave, also all reachable state have an invariant, and we can check if they satisfy the invariant, if they don't satify it the're not reachable.

## Invariant as useful for reachability

Still if we can decide, or compute in polynomial time the place invariance, spoiler: **IS POLYNIMIAL**
![[Pasted image 20230416172337.png]]

## Place invariant and boundedness

We're intrested in infinite tokens production they're bad, if we can find an invariant in a place, then for sure that place will be bounded!





---
# References

