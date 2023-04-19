Date: @today

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]]

# Trainng without finetuning

So the third big thing that happened after embedding and attention is in-context learning, maybe we can learn prompts, still using pretrained model without finetuning parameter but adapt the prompt to the `best` representation, so we assign a weights to our words.

![[Pasted image 20230417142217.png]]

So now we're training the words in the prompts and not the original model parameter.

![[Pasted image 20230417142245.png]]

## FLAN: Fine tuned language models are zero shot learners

Starting from T5 the full transformer architecture, the idea is to provide instruction and fine tune with respect to the instruction, where instructions are descriptions of the task we want to perform.

FLAN-T5 is fine tuned on a lot of tasks and it's being tested on a lot of benchmarks and `big bech`. This model gets significant better results with respect to smaller and non FLAN models.

![[Pasted image 20230417142837.png]]

### Ambiguity after fine tuning

Before finetune we wouldn't get the right answer, chatgpt has the same use of instruction tuning but that's just one technique used in the approach.
![[Pasted image 20230417143209.png]]

### Limitations

There are tasks that are open-ended : For example creative generation that have no right answer, or penalizing some errors more than other but all token level mistakes are equal..

### Templates

Hand crafted instruction with some specific data, FLAN for example has 137B of parameters, that is some cases is better than gpt3, sometime better than supervised model.

![[Pasted image 20230417143716.png]]


## Zero-shot chain of tought

Just provide the system with a task and exploit is knowledge to resolve a task, we can use few shot by giving model some example of what we want to get even better results.

Matemathical problems example (a): The model with few shot without chain of tought answer wrongly like when we're doing system 1 reasoning, mechanical reasoning is bad instead when using system 2 we're going with a chain of reasoning approach to explain to ourself what is happening. This is an example that we can see in the book [[Thinking, Fast and Slow]]

![[Pasted image 20230417143830.png]]

### Cheating with LLMs: Jailbreaking

Diverting system with inconsistent or incoherent prompt to make it produce something different from what it would produce.
![[Pasted image 20230417145449.png]]

Or make the model know that we're doing programming at higher level, by tricking it to writing googel header code.
![[Pasted image 20230417145535.png]]

## Language models are not aligned with user intents

As we can see there the model didn't do what the user asked for but aimed for a different thing, generating similar phrases.

![[Pasted image 20230417145715.png]]

Solution? `Finetune`.
![[Pasted image 20230417145807.png]]

### Emergent abilities

When model grow to bigger size they get abilities that we didn't suppose the model to have, there is no metric and no one knows why this happen.
![[Pasted image 20230417145929.png]]

As we can see in the graph the model has a jump very high from 20% to 40-70% in various tasks, this is something that we need to study and it's under investigation, why these model behaves better when we scale them. 

Sometime these model ***Allucinate*** they give an answer that they 'think' is correct (probability wise) but is really made up stuff.

## DataTuner architecture

Idea is that they provide some data that needs to be expressed in answer, the data is in form of key value pairs, and we want to explain data in a fluent, readable, understandable way and hasn't to contain data out from the input (no allucitating), so we train to be more compliant towards the original data, so produce information from the data and no more.

![[Pasted image 20230417150356.png]]

### Augmenting LLMs

Extending capabilities of chatgpt or other models like OpenChatKit, adding plugins like internet search, so what's generated is faceful and linguistically fluent.


---
# References

