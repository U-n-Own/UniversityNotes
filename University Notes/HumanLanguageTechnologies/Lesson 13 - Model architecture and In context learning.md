Date: [[2023-04-12]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]]

#Curiosity: Professor brought an easter chocolate egg of Transformers because today we will meet again transformer architecture

# Transformers architectures continued

Fine tuning is the right way to achieve good quality in specific tasks
![[Pasted image 20230412142940.png]]

We have different models for different tasks

## Again we evaluate models with GLUE

For example we want to se if something is logically derived 
![[Pasted image 20230412143018.png]]

## More benchmarks

Question and answers written by human, by reading an entire text and understanding then doing these query with answers that is the dataset used.
![[Pasted image 20230412143353.png]]

### More data, larger model better results

![[Pasted image 20230412143923.png]]

## What layers to use?

As we can see the last layers have a better representation of the data.
![[Pasted image 20230412144237.png]]

# More Architectures 

Bert and these architecture are surely intresting models, but people keep trying with different architectures like ***RoBERTa*** Robustly Optimized BERT pretraining approach, just more data and longer training this show improvements! 

Another model that come after was **XLNET**, introducing *relative positional encoding*, so instead of absolute position in the phrase "John ate the hot dog" we have a measure that is absolute attention and relative one, basically the relative say how much distant words are between each other.
Another intresting thing is *randomly permuting the other* for each training sentence, similar to masking, so permutation language modeling: next words is based on the ones before.

### Smaller models: ALBERT

A little BERT for self supervised learning of language. 

Introducing *Factorized embedding* parametrization, using small embedding size and projecting it to transformer hidden size with parameters matrix, albert is not faster than BERT but is smaller, much smaller.

### Exploring limits of transformers

By changing all the small thing that affect pretraining we can see what make the most
With extensive experimentation we could see how models changes.
![[Pasted image 20230412145155.png]]

### T5 

The results were pretty negative, best model had 11B parameters trained on 120B words cleaned. Exact masking and corruption strategies doesn't matter that much. But negative results aren't bad because make us understand better how these model changes with respect to the changes of architectures.


### Compute costs

![[Pasted image 20230412145439.png]]

Big numbers for computation required that are very expensive

More on training costs we can see
![[Pasted image 20230412145533.png]]

# In context learning

So far we've seen pretrained models in two ways

- Sample : from distribution they have learnt (providing a prompt)
- Fine tune: them on a task we care about predictions.

very LLMs perform some kind of learning without gradient steps simply from examples you provie within their context, and this results is very surprising.

This is called *in context learning* or *prompt engeneering*
![[Pasted image 20230412145945.png]]
More examples
![[Pasted image 20230412150124.png]]

## Distillation

Anothe approach to get smaller models because nowadays models are too big and costly to deploy, maybe are redundant and we can take part of the representation inside. 

*Distillation* try to compress models removing redudant part, like we compress files without losing information, several approaches is the **Teacher-Student** approach.

We train a *Teacher* use SoTA pre-training and finetune technique with max accuracy, then we take larg unlabeled data input with *Teacher* and train a *Student* on a much smaller size (eg 50x smaller) which is trained to mimic the teacher output, usually student objective function is MSE or Cross-Entropy

We can see 50k labeled examples and 8M unlabeled one and distillation works much better than pretraining and fine tuning with smaller model!

We can hypotesize why this works: maybe language modeling is the ultimate NLP task that generalize all the other subgenre we can say.

Training massive models learning millions of latent feature useful for other NLP tasks,  then finetune making the model biased towards a certain thing.

# Prompt based learning another breaktrough in NLP

After we say embeddings, attention we have another breaktrough in this field: Learning without actually "learn parameters", ***In contenxt learning***, which basic idea is give information to the model and it takes them and can explain with logical reasoning.

### PaLM

540B parameter by google understanding jokes and doing logical inference

![[Pasted image 20230412153717.png]]

### GPT-4

Large multimodal model that takes images and text input and produces text, exhibiting human capabilities and sometime is usually better than humans!

#### Practical challenges: large scale are costly

Model tuning is costly because of the billions of parameters, we can use much smaller fine tuning data because of this model is very capable yet! 

### Zero-Shot learning

Also these model performs task without going to be finetuned : Zero-shot

For example describe a task: "Translate a English to French", this is the prompt and the LLM manage to satisfy very well the taks, no finetune, and what if we provide few example in the current context? Well: even better model, but without changing or doing gradients steps!

Also with trivia we can teach the model with only prompting, not even finetune, in fact if we do a One-shot or Few-Shot we can outplay finetuned model this is rather incredible!

Most importantly improvements doesn't plateau they can still get better.

One question arises: if the model is better only by prompting, is there a better prompt or there is a better wording of prompt that makes the system better?

Prompting can make a difference, can be given sub optimal prompts for example.

- X is located in Y?
- X is located in which country? Y
- X is located in which country? In Y

These three prompts are one better than another and the model is capable to get a better grasp better the prompt.

### Learning prompt and prefix tuning

Can we do steps of gradient descent to learn the best prompt or make models to self prompt the best.

In order to perform the task with prefix tuning we add some words in front of the question that's the prefix, the prefix tuning will be handled from other weights tuned separately just for this. 

Prompt tuning: model stays frozen pre trained and we have new weights that are combined to learn new representation of current context that is much better

![[Pasted image 20230412155827.png]]

Another variant is task prompts for three things A,B,C then we take the best prompt for certain data and we train this way.

Next time we will see intructions for making the tasks!





>[!info]
> 






---
# References

