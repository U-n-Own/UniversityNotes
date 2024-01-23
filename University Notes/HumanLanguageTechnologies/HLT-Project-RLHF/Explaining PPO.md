
```
---
share: true
---
```

[[RLHF]], [[Transformers]], [[LLMs]], [[Reinforcement Learning]], [[PPO]]


## The recipe for RLHF


1. Pretrain a Model
2. Train a Reward Model (RM) + Gather data
3. Fine tune with RL

If one fine tune the initial model on some data they prefer to have it's called *Preference model pretraining*.

	Then we use the model to generate data to train the RM, this model takes text and output a reward signal, we go by sampling 'prompts' from a dataset

## PPO is Actor-Critic

We saw PPO briefly in [[Lesson 20 - Policy-Based RL#Proximal Policy Optimization]]

So is an hybrid (A2C), actor: policy-based method a critic: follows value-based method. 
With PPO we aim to stability and avoid larg policy changes, restraining them in $[1-\epsilon, 1+\epsilon]$.
### A glance to what we're maximizing

![[Pasted image 20230909181314.png]]

We do Gradient Ascent of this function to get higher rewards (expected: cumulated) and avoid harmful actions, if the step size is too small training becomes slow instead for big jumps training becomes unstable. Advantage function i discussed here [[Lesson 20 - Policy-Based RL#Estimating the advantage function]].

## PPO step by step

![[Pasted image 20230909183748.png]]

First we need to understand what are those strange symbles

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}(a_t|s_t)}}$ is the **ratio function** if this is bigger than $1$, the action is more probable in the current policy than the old one, the other way around if is between 0 and 1.
- Unliclipped part: why multiply for $\hat{A_t}$? First, the ration can take place of the log probability in the preceding objective function, second and more importantly: because Advantage function will make this positive if good action is taken otherwise will penalize, but if the action to be taken is **much more probable** in the current policy this will lead to significant policy gradient steps and violate the proximality we want, so we have to clip it in a range. This can be done with two ways: PPO or TRPO (uses KL divergence constraining policy update if KL is small enough taking more computation). PPO is good because directly clip what we want (in the original paper $\epsilon = 0.2$)

## Visualize PPO from the paper

![[Pasted image 20230909185223.png]]