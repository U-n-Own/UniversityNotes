Date: [[2023-04-18]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Reinforcement learning basics

Learning from experience and interaction is the foundal idea at the base of RL.

What changes in RL other than other paradigms and what makes it peculiar that make we treat it in a different way.
The way these model approach learning is goal-based, we're trying to reach a goal.

The agent can have a model of the world, if that is the correct model with full information we can know a prior what is the best action, but if the agent change the world in an unpredictable way become difficult, sometime we don't have the world model and we need our agent to explore it.

*Value function* and *policy gradient* are used when you don't have a model of the word.

## What characterize RL?

No supervision also this ins't even a un-supervised approach in fact when speakign about RL we can consider it as a whole *new paradigm*, here we only have a reward signal, time matters (sequential data, continual learning), also there is a delayed asynchronous feedback, a thing that matter a lot is that when the signal arrive we have to compute the reward with a delay. 

*Question*: what happens in RL, if a signal delay the feedback for the reward to the agent?

*Answer$_{by\_GPT}$* :If there is a signal delay in providing feedback for the reward to the agent in RL, it can lead to slower learning and potentially suboptimal decision-making. The agent may struggle to correctly associate its actions with the delayed reward signal, which can lead to confusion and uncertainty about the best course of action. 

Another issue is stationarity: our time dependant process has a constant average, in reinforcement learning this isn't true, not only observational studies but also other, the agent is the cause of data drift, the action of the agant cause the thing to change.

And yet another important thing is the balance between *exploitation* and *exploration* our agent must explore some state to discover the good one that maximize reward and don't pick the bad ones, and then exploit that knowledge about the state to find a sequence of action trough the state that bring it to the goal.

#### Brief interaction with Neuroscience

Reinforcement Learning interact strongly with neuroscience and psycology, since this can be tought as the kind of learning that our brain do many concepts at the core of this paradigm are biologically inspired, also the psychological model of animals behaviour that matches empirical data, more on this can be found in chapter 14-15 of the books Sutton and Barto

### Reward

Biologically speaking we can see at the reward as the plasure and pain we're getting after choosing an active, the goal is to maximize pleasure. 

Is a scalar (negative or positive) $R_t \in \mathbb{R}$ a feedback signal, indicating how well the agent is doing at a given step $t$, the agent job is to maximize (cumulative, summing it up) reward. What we do is reward engeneering so find a certain reward function that works.
RL is based on the reward hypotesis: the thing we try to learn is based on the reward signal, if this isn't good then we're not learning.

Formalizing reward hypotesis this way

```ad-important
That all of what we mean by ***goals*** and purposes can be well tought of as the maximization of the expected value of the cumulative sum of a received scalar signal (reward)
```

Mathematically speaking (without discounting)

$$
Reward\_Hypotesis=max(\sum_{t=0}^n\mathbb{E}[R_t])
$$

Of course be careful to this, we want to give positive reward to our agent only if he does what we expect it to do, the rational action that bring it to the goal state, if we're playing chess and we can take a piece, but further we lose because we're taking the piece in that moment that shouldn't be rewarded. 

Rewards are only given when the "robot" does what we want to acheive, not how you want it achieved. To decide what is the good move is found in the initial policy and value function, giving to agent a *prior knowledge*.

### Sequential Decision making

The goal of an agent is to learn a course of action, a sequence, that make the agent earn the maximum cumulative reward, this means that we have a long term objective, not greedy so we could not have fast reward locally losing somethiging but obtainign an higher reward.
![[Pasted image 20230418112910.png]]
![[Pasted image 20230418190819.png]]

### Agents and enviroments

$S_t^e$ is the environment $e$ private representation at time $t$, $S_t^a$ the internal representation by the agent $a$.

- Fully observability $\rightarrow$ Agent directly observes the enviroments state 

$$
O_t = S_t^a=S_t^e
$$

But not always fully observable if this is not then the agent is `tricked` by the enviroment.
![[Pasted image 20230418113152.png]]

A simplyfing assumption is that usually we have fully observable enviroment, take for example the game of Pong, we're seeing the screen and we can get all the information, in a game like NetHack we have that the map isn't known until we enter in a room so that isn't fully observable.

#### Again a probabilistic view

Our $S_t$ is a Random Variables (RV), under assumption of fully observability because if we know $S_t$ we don't need to know about the previous step so $A_{t+1} \rightarrow f(S_t)$. 

But this is a *markov assumption*: 
```ad-important
The probability for each possible value for $S_t, R_t$ depends only on the immediately preceding state and action $S_{t-1}, A_{t-1}$. This is best to be viewed as a restriction on the state and not on the decision process.
```

In fact we can say that this is a **Markov Decision Process**. The agent work on enviroment getting some experience of it and doing this build a model of the enviroment by exploring it. If we watch to the equality above we can say that $S_t^a = S_t^e$ if we have a fully enviroment we can know what action to take.

### Partially Observble Enviroments

POMDP formally is $S_t^a \neq S_t^e$, we can think to this like a Poker game, what we can do in this case is to have *Believes* on the enviroment, postulating latent variables from the observables.

![[Pasted image 20230418114103.png]]

## Components of a RL agent

What we can expect an agent is made of?

### Policy 

A policy $\pi$ is a set of model of behaviours of our agent, so what should our agent do at a given time. 
More formally a policy is a *mapping* from perceived states to actions to be taken in those states, in psycology is called set of `stimulus-response`. 

Our agent can have a *deterministic* (a lookup table) policy like this, where an action $a$ is taken over a certain state: $a = \pi(s)$
Or *stochastic* policy like $\pi(a|s) = P(A_t = a | S_t = s)$
Where each state will say how much probable an action is to be taken, there are cases in which continuous action are.

So policy is a *distribution of action given states*.

Learning the policy or something that leads to it is the objective.

### Value function

Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state.

How do we value the states? This function is a proxy for the cumulative reward we want to get, this value will tell us that we're in a good state or a bad state, so goodness of a state is a measure like a euristic.

If you're near a cascade then it's a bad state.
![[Pasted image 20230418115734.png]]

Formalizing :

$$
v_{\pi} = \mathbb{E}_\pi[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3} + ...  | S_t = s] 
$$

We're doing the expectation over a policy of (cumulative) rewards given the state.

$$
v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\big[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s\big]
$$

Where we have our expected values of a RV *given* that the agent follows policy $\pi$ and $t$ is any time step. Further we can express as we will see later the value of taking action $a$ in state $s$ under a policy $\pi$: $q_\pi(s,a)$: **Action-value function**

What's that $\gamma$? Regulating tradoff between reward we get immediatly now and those we will get in the future.

### Model

A *model* predicts what the enviroment will do next

Predicting next state $s'$ following an action $a$

- $\mathcal{P}^a_{ss'} = P(S_{t+1} = s' | S_t = S, A_t = a)$

Predicting next reward

- $\mathcal{R_s^a} = \mathbb{E}[R_{t+1} | S_t = s, A_t = a]$

The expected reward after we executre an action in a specific state, driven by the previous probability

What assumption we're using? Well conditional independance assumption: $P(S_{t+1}, R_{t+1}| S_t, A_t)$, can be factorized in this $P(S_{t+1} | S_t, A_t)P(R_{t+1}| S_t, A_t)$.


## Maze environments

We have different state we can be and some are better other are worse, this maze is our enviroment full observable. States in this enviroment are position on the grid, where agent is.
![[Pasted image 20230418115800.png]]

What's the policy?
For each white box we have a single action we can take, a stochastic policy could be to have all the four direction and weight those to a certain probability to happen.
![[Pasted image 20230418120235.png]]

Visualizing expected cumulative reward.
![[Pasted image 20230418120441.png]]
What is the model of the enviroment? Well an an agent can see all the maze knows it exactly where to go, the walls and has a perfect model of enviroment, but we can get a partial model of our enviroment, like this. Maybe we can even don't know the reward of some states.
![[Pasted image 20230418120653.png]]

## A Taxonomy of RL agents

If the agent has a model of enviroment we're *model based*, if the learning is focussing on value function then we're in *value based*, if we're learning a policy we're *policy based*, there can be model free where we only learn policy and not model, *actor critic* is when we're learning policy and value function.
*Model free* methods are those that learn by trial and errors workign the opposite way to the planning we have in model based.
![[Pasted image 20230418120736.png]]

## Learning vs Planning

Initially the enviroment is unknown the agent interacting with enviroment learns it improving is policy. If we have a model and we can simulate what are we gonna do we're planning by simulating the game of chess for example and we can go down in the promising paths. The agent improve policy by planning

# Markov Decision Process

Former model for enviromentm, MDP formally describe an enviroment for RL, these are fully obs. so the current state completely characterize the process. So a MDP is basically a *Markov Chain* with some additional stuff. Usually we have sets and some distributions. The set of states and action can not be only finite, can be infinite sometime. Then we have the usual state transition, triggered by execution of an action (we saw it in [[Lesson 7 - Hidden Markov Model]]).

![[Pasted image 20230418121919.png]]

## Return 

Sum of discounted reward that goes from $k=0$ to $\infty$, is a serie! This $\gamma^k$ gives us something decreasing if the parameter is less than $1$. Why adding **discount rate**? Well we need that serie to converge

For different value of $0 \leq \gamma \leq 1$, we have different behaviour.

- $\gamma \lt 1$ the infinite sum $G_t$ converge as long the sequence is bounded.
- $\gamma = 0$ the agent is *myopic* in being concerned only with maximizing immediate rewards, so choosing $A_t$ to maximize $R_{t+1}$ without watching further in a greedy way.
- $\gamma \approx 1$ the return objective takes future rewards into account more strongly, the agent has a farsight.

![[Pasted image 20230418122132.png]]

For a reward of $+1$, we can solve the infinite serie and get that 

$$
G_t = \sum_{k=0}^{\infty} \gamma^k = \frac{1}{1-\gamma}
$$

## Bellman Equations for MDPs

Things that are relevant are value function, we want to derivate a formulation of value function, that is the *expected cumulative reward*, where $G_t$ is the **total discounted reward** we saw earlier.

![[Pasted image 20230418122249.png]]

From the book:
![[Pasted image 20230419114519.png]]
"Note how the final expression can be read easily as an expected value. It is really a sum over all values of the three variables, $a, s_0, r$. For each triple, we compute its probability $\pi(a|s)p(s',r|s,a)$, weight the quantity in brackets by that probability, then sum over all possibilities to get an expected value." Equation $3.14$ is the Bellman Equation for $v_\pi$.

We can leverage the case in case $k=0$ and what we get is the expectation for rewards at $t+1$.
The second part with the $\gamma$ is the exectation of the value that you get as soon as you reach the next state. So we can see that $v(s)$ is expressed in term of itself, who defined what state i can reach? Policy function, all this will become expectation under the policy.

So expectation of reward (expected reward) in a specific state is our $\mathcal{R_s}$ and then we can see the example of maze for the next, if we have more than one state we can transite we can say that we have a probability to do the transition and we want to evaluate also the reward we get when we do the transition.

A different view of bellman equation. We're taking the max value over all the action of $R(s,a) + \gamma V(s'))$ becausewe want to find the optimal policy that maximizes the expected total reward over time. As we will see later.
![[Pasted image 20230418122628.png]]

### Value function

![[Pasted image 20230418123240.png]]

The action value function will tell instead to "how good is to be in a certain state", also "how good is to be in a certain state and doing a certain action".


### Bellman Expectation Equation – Value and Action-Value Functions


![[Pasted image 20230418123534.png]]

These two things are writing $v$ in terms of $q$ in the first equation and the otherway around in the second one.

Now we'll gonna pick the 

$$
\sum_{a\in \mathcal{A}} \pi(a|s)q_\pi(s,a)
$$

And put it into the $v_\pi(s)$

### Value and Action-Value Functions – One More Step of Nesting

With a certain probability we execute and action, then we evaluate the expected reward of that action and then we will weight all the state we can reach if we exectute $a$, the first summation pick one action, and the second instead sums over all the possible arrival state, weighting it by the probabilities, some state are not reachable due to constraint, some of them have probability of $zero$ exploring all the possible arrival state, and asking to each of the state the value functions and averaging them sum them to the rewards. All this we will do in a single step lookhaead.

For $q_\pi(s,a)$ instead we're doing it by just getting expected reward for that state and action and we take another step to assess how good is that action in that next state. But if things are not observable we marginalize *over all possible next states, weighted by their probabilities under the current policy. This gives us the expected* -> Is this true?
![[Pasted image 20230418123921.png]]

### Finding optimal policy

Since to solve RL tasks means finding a good policy, how can we get a policy now? Or better, how can we get the best policy?

We know the $q$ the optimal policy is the greedy one. 

![[Pasted image 20230418124635.png]]

What if we don't have $q_*$? We can still get an optimal policy

### Bellman Optimality Equations

Trading expectations with maximization.
![[Pasted image 20230418124749.png]]

### Iterative Policy Evaluation

So now we can solve this problem in an iterative way.
Let's assume we have a policy, and assume a recursive formulation as an incremental formulation. For all the states in parallel.
![[Pasted image 20230418125308.png]]

The first sum is when we choose the actions $a$ weighted by their respective $\pi(a^{(i)}|s)$.  The second sum is the setting of the arrival states.

This is called backup diagram, btu why, these diagram relationships form the basis of the update or *backup operations* of our systems.
What the diagram signifies is tht we have the solic circles (our couple of state action), starting at $s$, the root node the agent can take any of the bottom actions (based on it's policy). 
The Bellman Equation averages over all possibilities, weighting each by it's probability of occurring. 
![[Pasted image 20230418125829.png]]
### Take home lesson

```ad-summary


```


---
# References

