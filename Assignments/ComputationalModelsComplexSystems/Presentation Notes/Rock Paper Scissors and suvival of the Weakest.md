Notes take from [this paper](https://royalsocietypublishing.org/doi/pdf/10.1098/rspb.2001.1670)

In rock-paper-scissors game the stategies form a cycle, usually cycle are rare in competitive plant in marine sessile organism are quite common. In the paper we consider three species in a competitive loop, this simple ecology exibhits two counterinteractive phenomena:

1. The species that is least competitive is expeted to get a larger population and when there are oscillation in finite populations to be the least likely to die out.
2. Evolution favours the most competitive individual within a species, which leads to a decline in population

## Principle of Competitive Exclusion

```ad-quote
At equilibrium, the number of competing species that can coexist is no greater than the number of limiting resources
```

So how biological diversity is mainteined?

We can think to enviromental noise that prevents reaching of equilibrium, given a disturbance level we may even promote diversity. If external influences are unimportant, intrinsic dynamics may keep away from equilibrium, allowing coexistance of different species.

In the paper the competition is for "space", usually in ecosystem there are a lot of interacting species with complex network of *competitive relations*.

We treat the simplest non-trivial cyclic network with three species with relationship analogous to Rock-Paper-Scissors. It has been shown that cyclic competition  may be dynamically stable.


## Mean Field model

```ad-note
Cycle is $r\to s \to p\to r$
```

Let's setup the mathematical model of our word let's call $\mathcal{N}$ the word with available sites, each site can be occupied by three species $r$ (rock), $s$ (scissor), $p$ (paper), which occur in proportion $n_r, n_s, n_p$.
With this constraint ($n_r+n_s+n_p=1$), at each time-step two random sites are chosen.

The rules as as follows

1.  Occupant on first site can occupy a second (presuming is not taken with a probability)
2. $r$ can invade $s$ with a probability $P_r$ the same for the other two according to the cycle with ($P_s$ and $P_p$) all the other invasion probabilities are zero.

The change in population is due to the proportion of each species in the total population and the rate of change is modelled by the mean field equation

For the species $r$ for example will be
$$
\frac{dn_r}{dt} = n_r(n_sP_r-n_pP_p)
$$

Similar equation for the other two species. A unit of time $t$ is $\mathcal{N}$ individual timesteps and it's called epoch, later we will define our big is the space.

Dynamic is invariant to rescaling of invasion rate and $t$, the population densities forms fixed points of the mean field equation, we can obtain by setting the rate of change to zero.

In the following image we can see the dynamics onto a simplex

![[Pasted image 20240629155728.png]]

Calling $R$ the density at fixpoint of specie $r$, follows that

$$
R = \alpha P_s 
$$
$$
S = \alpha P_p
$$
and 
$$
P = \alpha P_r
$$

where $\alpha = (P_r + P_s + P_p)^{-1}$

Discussing now of dynamics of population the rate of invasion of a population is not quantified as it is but as "how much of the other population can i invade?" ratio, so the most aggressive species has never the higher amount of invididuals as we were saying at the start, so smaller the species more aggressive and competitive it is and bigger it is the other way around behaviour will emerge from that so this is reflecting clearly the point 1 at the start. And point 2 says that to get more competitive we want to follow the point 1 so get a decline in population and thus the cycle continues because in the future we will get favoured by this evolution strategy.

This is result of cyclic nature of our system, there is an odd number of species in the competitive loop, lowering invasion rate of one leads to a descrease in it's population and thus lower also the invasion rate of the population that invades it. 

*Lowering the invasion rate of a species promotes the growth of it's population*

In a finite population the trajectory of the density in our system will tend to overshoot the orbit eentually leading to a species becoming extinct.

The quantity $\lambda = (n_r/R)^R(n_s/S)^S(n_p/P)^P$ is invariant along each orbit with $\lambda = 1$, when we reach a fixpoint lambda goes to zero since we multiply (one or more go extinct).
Can be shown that if $r_{min}$ is the minimum values of $n_r$ along an orbit then

$$
\partial r_{min} / \partial R\|_{\lambda = \text{constant}} >0
$$
The specie with smallest fixpoint density has also the lowest population along any orbit. If invasion rates are unequal then the species with lowest fixpoint is the one that more likely will go extinct, while the species that survives is then the one that has the lowest invasion rate and thus will take over and win, this comeptitive system there is paradoxical survival of the weakest.


## Lattice Model

If dispersal is local rather than being long range then the dynamics is changed, while a species may become locally exstinct, distantly subpopulations oscillate with unrelated phase and on a large enough domain global extinctions are unlikely. (Very little probability to take out the last one). In the experiments the paper uses a square lattice as the model big $\mathcal{N} \times \mathcal{N}$, at each time step the second individual is randomly chosen to be one of the eight neighbours of the first (taken randomly).

### Behaviours under different invasion rates

Fluctuations are stabilized with spatial distirbution depending on invasion rate.
1. if the rates are equal population form lcumps with maximum size of 100-1000 individuals, for unequal rates there is a variety of spatial structures.
![[Pasted image 20240629163638.png]]

1. If one of the invasion rate becomes much larger than the other two, then the two species form disconnected islands amongs the third

![[Pasted image 20240629163704.png]]

These diagrams are very similar to the ones of Percolation and in fact i found [this study](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f431fccce261294e4910be050b35f4a65fc5a6fa) linking this behaviour to percolation theory and self-similarity at a critical state transition scenario.




