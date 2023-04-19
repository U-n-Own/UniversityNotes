Date: @today

Status: #notes

Tags: #hlt, [[A.I. Master Degree @Unipi]]

# Convolution for text

How do we do convolution on text, or embeddings? Well in image we have a 2D convolution, instead with words we select a certain windows and slide the convolution across the three vectors. Convolving with a certain filter, and we get a single value for each group of words.
![[Pasted image 20230321093034.png]]

## Filters

Well filters dimension goes along with the word windows we take, so 10 words with a 3 filters and dimension 5, we get 18 outputs, 6 windows for 3 filters.

Filters have some parameters such as:
- Adding padding at start or end of document
- Size of the sweep step (how much is the jump)
- Possible presence of holes in the filter window

We can add more transformation to our window, like Max Pooling that take the max values from the columns
![[Pasted image 20230321093957.png]]

Why the model with word emebedding is doing worse? Well Embedding are taken also from test set, we want to have very good embeddings, then we do convolution on these embeddings, that could make lose information to these embeddings. 

First solution just concatenate the words and they apply a classifier, instead with the convolution is a bit different what we did. So convolution seems to work pretty bad.

## Sentiment Analysis on Tweets

We've seen that in sequence to sequence modeling as tagging convolution isn't that good, also translation is an example of sequence to sequence.
Sentiment Analysis, take a sequence of words 144 usually as max, and want to infer is sentiment are positive or negative given these phrases.

![[Pasted image 20230321102735.png]]

The CNN is good for this task it seems. Why? We can combine convolution and another technique: RNNs
![[Pasted image 20230321102940.png]]

### Distant supervision 

Idea is to use the CNN to learn better embeddings to refine them we can say. We can collect million of tweets containing positive and negative and then use this to further refine embeddings as sentiment **distantly supervised labels**.

Another approach

### Build sentiment specific embeddings

Sentiment specific word embeddings have two output, when word is positive and word is negative
![[Pasted image 20230321104115.png]]

## Ensembles of classifiers
![[Pasted image 20230321104235.png]]


### Results
![[Pasted image 20230321104315.png]]

## Sentiment classification from single neuron
In this net of 4096 units there was a single neuron that brought a lot of information, so that was like a "sentiment unit" 
![[Pasted image 20230321104430.png]]

>[!info]
> 






---
# References

