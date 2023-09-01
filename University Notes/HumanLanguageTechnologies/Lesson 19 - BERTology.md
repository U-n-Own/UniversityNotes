Date: [[2023-05-03]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]],[[Computational Linguistic]]

# Analyzing Large language models

Do LLMs really understand? What is the meaning of understanding something?

There are a lot of question to be answered on LLMs: philosophical questions like what is understanding, how public will take these new models what neural model tell us about language and many more.

## Linguistic Structure in NPL

How humans understand language? When we read we understand from the context and we can infer a lot, for example if we speak about a film in a good way and our roommate spoke about the film in a bad way we can understand that there is something that someone likes and other not.
Linguistic structure are described by discrete rules?

## What linguistic knowledge is present in LM

By finetuning we can represent really well a task and doing this LM reach state of the art, but can they do it without doing fine tuning?

Well, yes!

There is a paper by Rajasekharan Unsupervised NER, can do sentence tagging without doing fine tuning. LM has hidden word embedding for each of the word in vocabulary can use cosine similarity to see if they are close and build clusters. These are **Context Independent Signature**, and we see how they overlap, we can visualize the entity distribution we can retreive from clustering.


## LM effectiveness

***Entailment*** is a very nice property that emerges, we give the model a premise: "He turned and saw Jon sleeping in his half tent" the model infer what entails the sentence, doing this in the 95% of cases, is understanding? Or the model is cheating? 

We can see that it doesn't have a deep understanding of the text is fed in. Some experiments showed this, these model are using *simple heuristics* to get good accuracy instead of understading, they maximize for less errors and not for understanding, this happens to also show similar to 

```ad-important
Language Models minimize errors in the answering, they aren't deeply understanding
```

Entailment can be inferred by three methods: Overlaps

### LM as linguistic test subjects

![[Pasted image 20230503145121.png]]

- [[Word attractors]]

The issue here are the subjects, can LLMs handle these "*attractors*" in subject-verb agreement? An attractor is a word present that might misled the model to accept wrong tokens, for example `pizzas` and `are` both prural but we expect a singular article...
Right now isn't agree whatever more attractor degrades the accuracy.


![[Pasted image 20230503152325.png]]

### Breaking (Bad) models

![[Pasted image 20230503152510.png]]

### Analyzing interpretable components

Looking to the attention matrixes we can infer what type of knowledge we got into the model
Weights to pairs of words with attention head 1-1 we can't tell what is emerging.
![[Pasted image 20230503152704.png]]

Some attention heads seems to perform simple operations
![[Pasted image 20230503152801.png]]

Other seems to corresponds to linguistic properties, so `The` `former` `executive` these three words are really connected and make sense on their own.
![[Pasted image 20230503152855.png]]

Heads to coreference relationships `she-her`, `talks-negotiations`, and so on...
Words in different sentences that are written in different ways but refers to the same thing.
![[Pasted image 20230503153006.png]]


### [[Probes]]

Can we extract gramatical representation from setences? Parse tree inferring.
Extract information without tuning towards that, by just the simple having added more layers

![[Pasted image 20230503153223.png]]


Doing probing with increasingly abstract linguistic properties


### Emergent simple structures in neural nets
Embeddings again, we can see that words appears in the dimensional space as vectors.

We can give the sentence and give it to transformers and get embedding, we want to visualize it.
![[Pasted image 20230503153815.png]]

#### Syntax probes
We can exploit the parse tree in this way. Words are on the hyperplane but in different spots. So do geometric distances in the plane matter? Yeah.
![[Pasted image 20230503153934.png]]

##### Projecting phrases

We can build a parse tree by taking the minimum spanning tree in this vector space, we do this for each pair of words, usually in Parsing we take the maximum spanning tree, here we're saying that similar words are nearer to each other so we take the most similar in the tree structure
![[Pasted image 20230503154130.png]]


## Beyond word context

![[Pasted image 20230503154309.png]]

As we can see two answers are both ok with the first, but the second continuation is more proper to what we would do, so we would like to get everytime a setence that makes more sense than the others. To do this we need to capture the true meaning or understanding by context (need long dependancy).
For example here the chef isn't going to leave the resturant and go in another one, but would buy the food to replenish his own resturant.

### Transformation approach

![[Pasted image 20230503154528.png]]
![[Pasted image 20230503154557.png]]

$d(w_i,w_j)$ is the number of arcs, by transforming embedding we have a way to construct a parsing tree
![[Pasted image 20230503154716.png]]

### ChatGPT is often wrong?

No, if people seeing chatgpt do an error and conclude that isn't good then aren't really experimenting, summary of results says that chatgpt seems to performs better than zero shot learning on a lot of tasks. Also sometimes the prompts aren't good.
![[Pasted image 20230503155206.png]]

### Conclusions 

![[Pasted image 20230503155540.png]]

>[!info]
 






---
# References

