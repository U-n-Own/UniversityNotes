Date: @today

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]]

# ChatGPT

This was one of the application deployed and most used in a short time in the entire history, hundreds of millions of users in a very short time.
![[Pasted image 20230417152235.png]]

## Previous version as basis GPT-3
The new model is just a version of old models but trained with the help of **Reinforcement learning**.
RL is used to teach the model to engage with people in dialogue, so training the model to have interaction with a story so a dialogue that reach a certain goal in a conversation, in this case satisfy the requests that user gives. This is what RL is used for and we will see how it works later.

![[Pasted image 20230417152342.png]]

### Training of ChatGPT
![[PASTED IMAGE 20230417152514.PNG]]

### Criticisms and limitation 

People try to put chatgpt to the limit by giving bad questions, or asking a lot of question until they get a mistake and then point out that is a *stochastic parrot that amplify biases in data*: meaning that they pick sentences in the data and spit out what they was trained on (in reality they do not copy training data) these models are ***generating*** every time new data, moreover Chomsky said that LLMs will never be able to do logical reasoning, but he was wrong. Also the definition of understanding, if we want to know if someone is understanding we asses this person with tests, so we can say that GPT is able to understand, not in the same way of how humans do, but it's another thing. 

Lack of compositionality refers to the fact that the meaning of a whole sentence or phrase cannot always be predicted by the meanings of its individual parts. This can occur when words or phrases have idiomatic or metaphorical meanings that are not transparent from their literal definitions. For example, the phrase "kick the bucket" means to die, but this meaning cannot be derived from the individual meanings of "kick" and "bucket." Lack of compositionality can make understanding certain expressions difficult for non-native speakers and can also lead to ambiguity in language processing.

Also text is not sufficient as source of knowledge: we experience the word with no only text or reading but by seeing, hearing and a lot of other things. So yeah text is not enough but in books we have a lot of examples of what hearing is and what seeing is, so maybe these models are assuming these knowledge without even having it.

### Lack of logical reasoning

These model sometimes fails to reason, but they have logical abilities. Here a trick logical question.

![[Pasted image 20230417154253.png]]

Askign chatgpt with different prompt: A logician reasoning like, we don't know if jack is married to Ann but what if we assume they are married in a case and not married in another? Well we can infer that there is a solution by pure logical resoning.
![[Pasted image 20230417154501.png]]

## Reinforcement Learning

Old thecnique used in old Artificial Intelligence, use a lot in game play, we have a goal: win.

We have action that players can do and each action will bring the game in a new state, we want to reach the winning state trough a sequence of action, usually we have another player we assume is again us and is a player that is playing rationally, so everytime will execute the best action to maximize it's reward.

In the case of LLMs suppose we want to be able to summarize well. We have two possible summaries, we can assign a certain reward to each of them to assest the goodness of the summary

![[Pasted image 20230417155009.png]]

### Some mathematics behind RL

***Maximizing*** expectation of the score (reward) is what we want

![[Pasted image 20230417155113.png]]

**Log-derivative trick** in Reinforcement learning consists in taking the logarithm of a policy and differentiating it to obtain the gradient of the policy with respect to its parameters. This trick is commonly used in policy gradient methods, where the goal is to find the optimal policy that maximizes the expected reward. By taking the log-derivative of the policy, we can express its gradient in terms of the advantage function, which measures how much better a certain action is compared to others in a given state. This makes it easier to compute and update the policy parameters using stochastic gradient ascent. The log-derivative trick has proven to be a powerful tool for training deep neural networks to learn complex policies in high-dimensional state spaces, where traditional reinforcement learning algorithms may struggle. It also allows for more efficient computation of the policy gradient, reducing the number of samples needed to achieve good performance. Overall, the log-derivative trick is a key ingredient in many state-of-the-art reinforcement learning algorithms and has contributed significantly to the success of deep reinforcement learning in recent years. 

![[Pasted image 20230417155224.png]]

Reinforcement learning is good because of what we're doing, enforcing good action to happen more and doing this penalizing the bad ones.

![[Pasted image 20230417155438.png]]

### How do we model human preferences

How to teach the model that is doing good? Using humans to annotate, give score for a certain summary and so on. This is good but not reliable, some people can say positive reward is a certain number and other can be of another line of tought
![[Pasted image 20230417155553.png]]

### Putting all togheter

Starting from a pretrained model, we add a reward model and use this to train parameters with human feedback, now we do Reinforcement Learning so updating weights to a certain reward and the formula at the bottom is the combined basic score from reward model and the fact that we have a difference from correct and wrong answers.

![[Pasted image 20230417155828.png]]

### Improvements with Human feedback is better than humans 

![[Pasted image 20230417160022.png]]

### Again chatgpt training

![[Pasted image 20230417160141.png]]

#### Collected tasks
![[Pasted image 20230417160254.png]]
>[!info]
> 






---
# References

