Date: [[2023-04-28]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# Policy gradients

Last time we parametrize a value function now we parametrize a policy, approximating the policy itself.

In actor-critic we have both policy and value function.

![[Pasted image 20230428142658.png]]

We've seen simple cases from linear regressors to Deep Q learning (atari paper).
What is the problem? If i'm trying to giving score to my states or value function we can have a very big action space, this is one of the reason, what we do is put our $\theta$ thing, near the policy, so parametrizing that. We're seeing the model free learning, the model based is too advanced for this course.

### Advantages and Disadvantages

- Better convergence properties: if you do bootstrap if difficult to get convergence specifically for SARSA and Q-learning, with this policy gradient we have better convergence properties, but it's not easy to get to convegence (Higly sample inefficient)
- We can learn stochastic policies: until now we've seen that policy learn is direct consequence from value function (greedy one), action is deterministically choosen given value function we've introduced $\epsilon-greedy$ policies to add some stochasticity, but that's is only to optimize and discover more about the word, instead here we're learning in a stochastic way, not only by exploring and using it.

---

- Typically converge to local optima
- 

## Example with Rock-Paper-Scissors

We cannot be informed about the action that our adversarial, so in this case a randomized policy is the best, because any complex determistic policy can be exploited and beaten by an agent, unless you're sheldon and use Spock that beats all.

![[Pasted image 20230428143442.png]]

## Policy gradient

Starting with an assumption in an episodic words.

Assessing a policy easy: What is the value at episode one? If the policy is good we get higher value, this is measured by the expected cumulative reward obtained by following the policy over a fixed period of time. If the policy is effective, it should lead to higher cumulative rewards

![[Pasted image 20230428143806.png]]

Continuing enviroments are more life similar: We spawn we learn trought the time, and to do this we need something that is not a single state that summarize our life, so we need to assesst value of certain state, certain states those visited more often, so taking expectation with respect to the state, expectation of the value to be in each possible state, that contribute. But $d$ depends on $\theta$ going in certain state is influenced on the policy because bad policy will bring us more likely to not the optimal or worst state where the value is lower, so we want to maximize that.
The *average reward*, is the expectation of the reward we can get, states are drawn over stationary distribution, and the states are drawn from the policy. 

These function can be used to estimate quality of our policy.

```ad-important
Policy Gradient Theorem

```

![[Pasted image 20230428144849.png]]

Gradient of the log of a distribution is the *score*

## Score function

Usually $\pi$ is a distribution from a family of exponential

When maximizing the log the trick is that P(S'|S) x P(S0) x P(S | a), only the last one is dependant because this becomes a sum,

![[Pasted image 20230428145044.png]]

## Softmax policy

Now we get our $\phi$ giving us a $k$ dimensional vector multipling it by another vector $\theta$ that is a scalar and we take the exponential of that
When we want to search for new value of $\theta$, we use the old value + $\alpha$ (learning rate) $\nabla$ gradient

![[Pasted image 20230428145352.png]]

So the term $\phi(s,a)$ is the term that we get (exponential go away because of log), (specific feature for specific fucntion), and minusing the expected feature over all the possible actions

## Gaussian policy

![[Pasted image 20230428145826.png]]

We want a distribution giving us continues value: a gaussian. How do we make the mean of our gaussian a good represetation for our policy?

Then we draw sample from our gaussian policy and taking derivative of gaussian distribution using theta to estimate, before we had with the expected feature over action and now we parametrize over the mean of the gaussian

## REINFORCE

Since $v_t$ are unbiased samples we are doing the correct things.

![[Pasted image 20230428150827.png]]

## Policy gradient vs Maximum likelihood

Given these photo we want to guess correct steering angle, so the policy-gradient we're taking is that for monte carlo, an average over episodes, then we unfold trough time time istant by time istant (inner sum) and compute policy gradient over episodes.

At the right hand side we have a maximum likelihood, we're outputting a log probability, the only thing that differs is the value function, so looking very much like maximum likelihood but adding a component.

![[Pasted image 20230428150903.png]]


## PG Vs ML - Gaussian Policy 

More frequent action state are not estimated how good these were state (action in state), maximum likelihood is intrested only in frequency, in Policy gradient we're weighting, so more than a fully frequentist approach is a trial and error system. Parameter move in the direction were action lead to good reward and do not move toward bad reards action states.

![[Pasted image 20230428151307.png]]

## Policy gradient is on-policy

![[Pasted image 20230428151651.png]]

Learning from actual thing happening in real life is not effective, this run on policy in real word, humans lives are costly too soo this is inefficient.

![[Pasted image 20230428151753.png]]

We've high variance in this case with

## Actor critic: Reduced variance using a critic

Instead of $Q^{\pi}$ we get $Q_w$, we get two function approximator, the actor: function that approximate a policy but the actor is using the policy gradient to learn using the critic (a neural network approximating the function).

Actor take an action, critic says: You're doing very good.

Doing this in parallel of generalizing policy iteration: policy update with policy gradient and policy evaluation by value function approximation.

![[Pasted image 20230428152953.png]]

We're gonna use TD learning because MC is as we said is inefficient variance wise
![[Pasted image 20230428153318.png]]

## Action-value Actor-critic

We're doing like Q learning where we have a bucked we pick samples and put them in the bucket and sample from there we need also to pick new sample from the future because we're doing TD learning, the $\delta$ as usual is the 
![[Pasted image 20230428153706.png]]

![[Pasted image 20230428153903.png]]

## Reducing bias using a baseline

Substracting the average values to be in all the states, not that different by what we do in bias in neurla network, bias capture the average activation of synaptic potential, because without that we would be learning also that, we can caputure the average with that bias.

If you baseline fucntion is only dependent on the state, than we take expectation of policy gradient and if that is zero we're not changing, expectation is a linear operator, the derivative doesn't apply to B(s), and since it doesn't depend on the policy the can rearrange like that, gradient of something with all fixed on $1$ is zero. So not biasing our estimation. 

So we can substract the value function our baseline, so Q-V we already met it, the *Advantage function*, telling that in some state we don't want know what happens when travelling in state we're only intrested in choosing action. Plug in the advantage function in policy gradient in place of Q function we have an unbiased estimate.

![[Pasted image 20230428153947.png]]

## Estimating the advantage function


We can introduce the TD error an unbiased estimator of the advantage function

```ad-important
Our $A(s,a)$ is an indicator of what? If this is greater than 0, Q was bigger than V and so this action is better than the other action in possible at that state, if is negative then V was bigger this tell us on average every action taken in that state is worse than the expected return from that state, in other words is better not take any actions!
```

![[Pasted image 20230428154606.png]]

Expectation of TD error is the advantage function.

![[Pasted image 20230428154715.png]]

Plugging in the TD learning erro into the policy gradient is a good idea because at this moment our exectation is all a chain of unbiased estimators.

## Natural policy gradient

We want thing that are good under sample aspect, parameterizing a policy in a space we can have two actions, our $\theta$ can be changed and a slight change can do a very big change in policy, in an heavy way, this is really difficult introducing instability in learning.
Changes made to parameters needs to be controlled.

![[Pasted image 20230428155704.png]]

Natural gradient tells us how much to trust each direction of the gradient, because we don't want to fall

Have we a measure to define how curved is the geometry of our function yeah is the *Fisher information matrix*

![[Pasted image 20230428155842.png]]

This is a computational nightmare, so what we do? Let's do it locally

## Trust region policy optimization

Confronting the policy: new and old.
How do we do it? Kullback Liebler divergence where the divegence is bounded over a certain $\delta$, this is constrained optimization having costs, starting from this problem, but instead of solving this we reformulate it using a series of approximation that tell in the end what we need to do pick an $\alpha$ coming from that definition above. The denominator give us how much to trust the region, rather than optimizing that function we're optimizing a lower bound, instead of do it freely, we're doing it in a constrained optimization setting.

Pick a region of the space and within the $\epsilon$ values we're getting the right optimizer and

ICML paper 

![[Pasted image 20230428155939.png]]

## Proximal Policy Optimization

The thing above is second order optimization, now we want approximation of approximation

![[Pasted image 20230428160649.png]]

the clip thing is measure the ration between old and new policy, if ration is too big, clip it in the interval 1-$\epsilon$ and 1+$\epsilon$. How this comes out is important, this is the de facto standard if we want to do policy gradient learning: USE PPO

### Take home lesson


Intoduce a learned value function, allowing to control variance and work in model free, we can learn in real time, replay buffer and so on.
Reducing variance and introducing bias, so we need to be careful to what function we use, estimating advantage function (unbiased estimator). Can we combine this with n-step and eligibility trace, instead of using TD-0 use TD-$\lambda$





```ad-summary


```


---
# References

