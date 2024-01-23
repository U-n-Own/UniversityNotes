[[RLHF]], [[Transformers]], [[LLMs]], [[Reinforcement Learning]]

## The recipe for RLHF


1. Pretrain a Model
2. Train a Reward Model (RM) + Gather data
3. Fine tune with RL

![[Pasted image 20230910180305.png]]


If one fine tune the initial model on some data they prefer to have it's called *Preference model pretraining*.

Then we use the model to generate data to train the RM, this model takes text and output a reward signal, we go by sampling 'prompts' from a dataset

---

# How PPO works in details?
##  Here for more : [[Explaining PPO]] 

---

# Fine Tuning with RL

Take a copy of our LM and use PPO




## Our strategy:

First we try to train a simple RM: NLP, SVM, Decision Tree..