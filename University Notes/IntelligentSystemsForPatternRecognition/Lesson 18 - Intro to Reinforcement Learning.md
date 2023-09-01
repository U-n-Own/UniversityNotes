Date: [[2023-04-18]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# [[Reinforcement Learning]] basics

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

Reinforcement Learning interact strongly with neuroscience and psycology, since this can be tought as the kind of learning that our brain do many concepts at the core of this paradigm are biologically inspired, also the psychological model of animals behaviour that matches empirical data, more on this can be found in chapter 14-15 of the book Sutton and Barto

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

> [!definition]
> A *value function* is can be tought as $\mathbf{E}$ *return* given a state

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

## Exploiting again Conditionally Independance

What assumption we're using? Well conditional independance assumption (assumiing that next state and reward are conditionally independant from the current state and current reward): $P(S_{t+1}, R_{t+1}| S_t, A_t)$, can be factorized in this 

$$P(S_{t+1} | S_t, A_t)P(R_{t+1}| S_t, A_t)$$

- Here, $P(S_{t+1} | S_t, A_t)$ represents the transition probability, which describes the probability of transitioning from the current state $S_t$ to the next state $S_{t+1}$ given the current action $A_t$. It captures the dynamics of the environment and the agent's interactions with it.

- Similarly, $P(R_{t+1} | S_t, A_t)$ represents the reward probability, which describes the probability of receiving the immediate reward $R_{t+1}$ given the current state $S_t$ and action $A_t$. It characterizes the feedback the agent receives from the environment based on its actions.

```ad-summary
This factorization simplifies the modeling and computation in reinforcement learning algorithms, such as value iteration, policy iteration, or Q-learning, where the agent aims to optimize its actions based on maximizing expected rewards.
```

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
### Value function and Action-Value function

The first is the usual value function under a certain policy, while we could further extend this concept by adding another information, the action taken. 

![[Pasted image 20230418123240.png]]

The action-value function will tell us instead of "how good is to be in a certain state", "how good is to be in a certain state **and** performing a certain action".

## Bellman Equations for MDPs

Things that are relevant are value function, we want to derivate a formulation of value function, that is the *expected cumulative reward*, where $G_t$ is the **total discounted reward** we saw earlier.

![[Pasted image 20230418122249.png]]

From the book:
![[Pasted image 20230419114519.png]]

Note how the final expression can be read easily as an expected value. It is really a sum over all values of the three variables, $a, s', r$. For each triple, we compute its probability $\pi(a|s)p(s',r|s,a)$, weight the quantity in brackets by that probability, then sum over all possibilities to get an expected value. Equation $3.14$ is the Bellman Equation for $v_\pi$.

We can leverage the case in case $k=0$ and what we get is the expectation for rewards at $t+1$.
The second part with the $\gamma$ is the exectation of the value that you get as soon as you reach the next state. So we can see that $v_\pi(s)$ is expressed in term of itself, who defined what state i can reach? Policy function, all this will become expectation under the policy.

So expectation of reward (expected reward) in a specific state is our $\mathcal{R_s}$ and then we can see the example of maze for the next, if we have more than one state we can transite we can say that we have a probability to do the transition and we want to evaluate also the reward we get when we do the transition.

A different view of bellman equation. We're taking the max value over all the action of $R(s,a) + \gamma V(s'))$ becausewe want to find the optimal policy that maximizes the expected total reward over time. As we will see later.

![[Pasted image 20230418122628.png]]



### Bellman Expectation Equation – Value and Action-Value Functions


![[Pasted image 20230418123534.png]]

These two things are writing $v$ in terms of $q$ in the first equation and the otherway around in the second one.

Now we'll gonna pick the 

$$
\sum_{a\in \mathcal{A}} \pi(a|s)q_\pi(s,a)
$$

And put it into the $v_\pi(s)$

### Value and Action-Value Functions – One More Step of Nesting

With a certain probability we execute an action, then we evaluate the expected reward of that action and then we will weight all the state we can reach if we exectute $a$, the first summation pick one action, and the second instead sums over all the possible arrival state, weighting it by the probabilities, some state are not reachable due to constraint, some of them have probability of $zero$ exploring all the possible arrival state, and asking to each of the state the value functions and averaging them sum them to the rewards. All this we will do in a single step lookhaead.

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

This is called backup diagram, but why? These diagrama relationships form the basis of the update or *backup operations* of our systems.
What the diagram signifies is tht we have the solid circles (our couple of state action), starting at $s$, the root node the agent can take any of the bottom actions (based on it's policy). 

The Bellman Equation averages over all possibilities, weighting each by it's probability of occurring. 

![[Pasted image 20230418125829.png]]

In order to obtain $v_{k+1}$ from $v_k$ this algorithm applies the same operation to each state $s$, replacing the old value with a new obtained from the $s'$ (successor) and the *expected immediate rewards*: **Expected update**.

Each iteration update the values of every state to produce new *approximated value function* $v_{k+1}$.

All update are called expected because are doing expectation over all possible next states.

![[Pasted image 20230707122020.png]]


### Evaluating iteratively in gridword

Imagine another game in a grid, we need to do a policy evaluation, and what policy we can give to the agent? A random policy, so each of the four action are equiprobable, then we evaluate policy on environment.

![[Pasted image 20230419162344.png]]

#### Evaluating policy

Initially our value is zero because we do not know nothing about the grid. Then we apply the *Bellman expectation*. Again summing over all action in the space (a term for each N.S.W.E. actions), so everytime we take action we pay $-1$. 

Note: In stochastic envirnoment if we take an action we could finish in a state that was not supposed to be reached, now we're assuming that we reach that state with our action in a deterministic manner.

In this case we can think to $v_k$

$$
\begin{align}
v_1 = 0.25[-1+0]^N+0.25[-1+0]^S+0.25[-1+0]^W+0.25[-1+0]^E \\
\end{align}
$$
Then we can see that keeping iterating we reach a stable point, all we do is to averaging the neighbour.
How do we change the policy in this case? Since we're maximizing total expected cumulative reward, at iteration 1 we want to reach the zero value and we have only 1 direction to get the best expected return go up or left for example, at iteration two we have a better policy that knows more about other states that previously used a random policy.
As we can see at iteration three we just reached a stable policy, in fact after that the value grewth only by magnitude. Learning policy directly converge sooner but it's more difficult to optimize. (We reached $\pi^*$)


![[Pasted image 20230419162527.png]]
![[Pasted image 20230419163753.png]]

### Improving policy

What we just did was get a random policy and change it with respect to itself evaluating with Bellman and improve it. Greedy approach pick one action that improves the policy (locally optimal).

![[Pasted image 20230419164502.png]]

### Iterating over policy

![[Pasted image 20230419164808.png]]

So giving the fact that policy iteration converge really fast we can do it in a single step.

![[Pasted image 20230419164855.png]]

This bring us to a **Generalized Policy Iteration**: Any policy and any policy improvement, and what are the best? Greedy one, we evaluate policy under greedy action, but now we're doing from Bellman expectation a maximization but incrementally, thing bring us to *Value Iteration*

![[Pasted image 20230419165047.png]]

![[Pasted image 20230707130452.png]]

#### Value iteration

![[Pasted image 20230419165224.png]]

![[Pasted image 20230419165209.png]]

In Sutton and Barton book we have the new backup diagram that change only an arc that signal the maximization step
![[Pasted image 20230419165354.png]]

## Wrapping up on  Model-based RL

Prediction: evaluate given policy
Control 1: Learning from a random policy and alternating policy eval and improvement getting to the best one
Control 2: We don't need to alternate, we need just to repeadetly update value function by doing maximization

![[Pasted image 20230419165517.png]]

# Model free RL

What happens if we get rid of the model?

We're moving from expectation based approach to a sample based approach, so montecarlo methods...

In fact now the complexity of enviroment is less, but now we pay that with the fact that we're doing stochastic update.
If i have sample of experience i can compute expectation as we can saw before in [[Lesson 10 - Sampling Methods]].

## Monte-Carlo Reinforcement Learning

We're trying to approximate the expected cumulative rewards: our $G_t$.

So a sample consist in a *Return* value, we can get a certain number of return and average them and our $v(s)$

Now we're **estimating value function** by sampling or better *experience*.

Monte Carlo methods solve RL, by averaging *sample returns*, for each $<s,a>$ pair

$$
v(s) = Avg(G_i)
$$

What we do is *incremental averaging*, learning our $v$ function after every episode.

Hence we want to learn value function from sample results.

The problem is that all episodes must finish, there are cases where our agent don't finish the episodes.

![[Pasted image 20230419170827.png]]

### Monte Carlo Policy Evaluation

Sampling episode of experience using the policy, after collecting one sample we use it to compute a new expected return

![[Pasted image 20230419170958.png]]

### Updating states

$G_t - V(S_t)$ comes from incremental mean, we can see it as moving current estimate of our function to the new estimate with a step size of $\alpha$

Where $N(s)$ is number of time I’ve seen state $s$.

If we visualize what is happening the current

![[Pasted image 20230419171235.png]]


## Temporal Difference Learning: TD(0)

This isn't the only way to do model free learning, in this way we can only learn at the end of the episode if they terminate, these are really strong requirements, so we need another way.

Learning directly from episodes of experience, every action taken is used for update and learn, but this entails that we're not using ground truth to learn, so we're not learning from actual return but with the reward of the current step.

Let's start from MC:

$$
V(S_t) \leftarrow V(S_t) + \alpha(G_t-V(S_t))
$$
This time we're changing the old value because what we get each time we do a move other than reward we get to another state and we can evaluate a *targer* the TD difference target, by using our value and we get an error $\delta_t$

$$
V(S_t) \leftarrow V(S_t) + \alpha(R_t +\gamma V(S_{t+1})- V(S_t))
$$
We have now a pseudo target we can change at each time step. We're doing ***Bootstrapping***

![[Pasted image 20230707182342.png]]


### Bias-Variance Tradeoff (again)


MonteCarlo policy evaluation is low to no bias, if we do the evaluate using $v_\pi$ but if low bias in MonteCarlo gets us high variance, while TD learning low variance and high bias.

MonteCarlo is sampling inefficient and we need to get to the end, so would take forever, instead TD is more sample efficient but has very bad convergence properties,

![[Pasted image 20230419172756.png]]

![[Pasted image 20230419173129.png]]

## Unifying and Generalizing

Monte carlo visualized

![[Pasted image 20230419173333.png]]

TD Visualized

![[Pasted image 20230419173347.png]]

Dynamic programming 

![[Pasted image 20230419173423.png]]



What if we use 1,2,3 action when updating TD learning, so allowing TD learning to execute more action, we're approxximating MonteCarlo, so TD-n when $n \rightarrow \infty$, is MC.

![[Pasted image 20230419173605.png]]

Now what if we use intermidiate estimate? Creating pseudo target $\{n,n-1,n-2...\}$, averaging them getting one target.
Putting some value $\gamma$ like a decaying factor of these step, now we pick $\gamma$ as a function that integrates to $1$ to get some nice properties. 

## $\lambda$-Return [Forward View]

In the forward view, the idea is to look ahead n steps in the future and estimate the return from that point, then average those returns over multiple trajectories. This approach requires complete knowledge of the environment dynamics and does not naturally support online learning because it needs to collect a batch of trajectories before updating the policy.

This $\lambda$ is a constant behaving like discount factor, pseudo reward far in time are discounted more. This is the ideal $\lambda$-return, if we stop at 1 ($\lambda=0$) we get basic TD-learning, at step two we get a TD learning averaged between the steps and at infinity becomes montecarlo($\lambda=1$).

But with this formulation we lose the possibility to do on-line learning, because depending on the $n$ we must take that many steps before updating. But this is not what we will do. How do we implement it? Well with **Elegilibity Traces**


![[Pasted image 20230419173846.png]]

The $\lambda$-return in the TD($\lambda$) algorithm is a method to assign weights to each of the n-steps taken during the current episode. It involves weighting the importance of each visit based on its temporal distance from the current step. The lambda parameter determines the decay rate of these weights, with smaller lambda values assigning higher importance to nearer steps. As a result, the lambda return is a weighted average of the n-step returns, where the weights are determined by the lambda values associated with each step.

So now the update is done by averaging over the $\lambda$-returns weighting the state

![[Pasted image 20230708173316.png]]

## Backward view TD($\lambda$): Eligibility Traces unifying MC and TD

As we were saying in a forward view we need to look forward in time to all the future rewards and decide the best way to combine them with the lambda returns, also the forward view needs until infinity (so we are doing MC), and by evaluating more steps we lose the **on-line learning**.

Elegibility Traces provide a way of implementing Monte Carlo in online fashion (does not wait for the episode to finish) and on problems without episodes.

Using eligibility traces: Frequency and Recency euristics.

Example: Mice getting electrocuted after hearing 3 times bell and then seeing the light, what triggered the zap?

- Suppose an agent randomly walking in an environment and finds a treasure. He then stops and looks backwards in an attempt to know what led him to this treasure? So closer locations are more valuable than distant ones and thus they are assigned bigger values, this bring us to recency and frequency as important heuristics.

![[Pasted image 20230419174349.png]]

Using the same logic, when we are at state s, instead of looking ahead and see the decaying return (Gt) of an episode coming towards us, we simply use the value we have and throw it backward using the same decaying mechanism.

We defined the TD-Error

$$
\delta_t = R_{t+1} +\gamma V(S_{t+1}) -V(S_t)
$$

We will backpropagate the error backwards with decay multiplying the error by eligibility trace at each state $s$.

$$
V(s) = V(s) + \alpha\delta_tE_t(s)
$$

So we saw that $v(s)$ is a vector of length $|s|$ number of state, that indicator will say to us : the vector with current state.

For example if there is a state visitate more often frequently and recently we assign most credit to that and we propagate reward to that state to also other states because usually to reach that state we come from other states. Backward responsability.

***The eligibility traces effectively capture the credit assignment problem***

![[Pasted image 20230708174154.png]]

Until now we've said how good policy is, now're going to learn it.

## On and Off policy learning

### On-Policy: Model-Free Policy Iteration Using Action-Value Function

![[Pasted image 20230419175250.png]]

The first one has the model but the $Q$ function is model free!

But before we need an exploration policy.

Learning on the fly experiencing we end in a local optima, because doing everytime greedy thing we end up everytime in the same spot, so we need to diversificate, adding a bit of stochasticity, we need randomization to avoid suboptimality.

### $\epsilon$-greedy Exploration

![[Pasted image 20230419175642.png]]

### Monte Carlo Control: Every episode

![[Pasted image 20230708181609.png]]


### On policy control with SARSA

What we will do is to start with an evaluation using TD learning, then select with $\epsilon$-greedy policy.

![[Pasted image 20230419175831.png]]


### SARSA: State-Action-Reward-State-Action (On-Policy) 

SARSA is a TD learning algorithm that want to learn the $Q$ function this time not the $V$ function! 

*So determine the quality of goodness to take a certain action in a certain state. So TD prediction becomes TD for control!*

Since is a MC hybrid we need to tradeoff exploration and exploitation and we have two version an On-policy and Off-policy, this is the on-policy one.

How to estimate $q_\pi(s,a)$? Recall that episode are a sequence of states and states-action pairs

![[Pasted image 20230708181940.png]]

### Why SARSA?

![[Pasted image 20230708182111.png]]

## SARSA algorithm

![[Pasted image 20230708182247.png]]

## Expected SARSA: Q-learning but with expectations

We're estimating the action value function by taking the expected value over all possible states not only by sampling one

![[Pasted image 20230419180202.png]]

## SARSA algorithm more specific

![[Pasted image 20230420143751.png]]

Eligibility trace is multiplied by $\lambda$ and $\gamma$ where lambda is how fast eligibility fades and the latter is discount factor. A reward discounted $k$ times in the past or future. $\lambda=1$ becomes MC.

## Off Policy:

We postulate two policy a target we want to optimize and the other is the behavioural policy the one we sample from, policy eval is from $\pi$ and policy learning is on policy $\mu$. 

- Learning by imitation: action provided by human and optimizing poliy of artificial agent, for a time evolution of our agent where policy changes, we're generating sample of experience.


![[Pasted image 20230420143941.png]]

When doing *Off-Policy* learning with action values we're doing ***Q-learning***

## Q-learning

The pseudo target is the red term : reward + bootstrapping term.

![[Pasted image 20230420144420.png]]

### Off policy control by Q-learning


![[Pasted image 20230420144650.png]]

![[Pasted image 20230420145000.png]]

Expected SARSA is doing exactly this, Q-learning performs a maximization instead SARSA perform expectation (Bellman expectation vs Bellman optimality).

### Q-learning algorithm

![[Pasted image 20230420145123.png]]

## Wrapping up: Dynamic Programming vs TD learning

The second row is the control part: we have Q-policy in model based, SARSA in model free


![[Pasted image 20230420145316.png]]



### Take home lesson

```ad-summary


```


---
# References

