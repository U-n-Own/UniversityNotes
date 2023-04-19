Date: [[2023-03-20]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]]

# Sequences

We will encounter a lot of sequences, sound, words and a lot of other type. HMM and Bayes are making a lot of simplification assumption, in HMM the semplification is that the model depends only on the previous states, these simple model are generative models that learn that distribution on the avaiable data and try to predict those maximizing the probability for example given the past examples we want to generate what type of word and what word will be (HMM), instead Naive Bayes pick the most recurring one based on data.

We want more **sophisticated models**, than those. These can forget old information that are not needed, but usually there are long term dependancies, so in order to go deeper in this representation to generate better output on larger ngrams. We will reach those that researchers call Transformers.

Problem that usually happens are on long training the `Vanishing gradients`, we have to be careful with this things.

## NN classifier

The NN classifier will produce a probability distribution for each tag:

   $P(t_{i,j}|w_{i-2}w_{i-1}w_iw_{i+1}w_{i+2}) = softmax(W\cdot [E(w_{i-2})...E(w_{i+2}] + b))$

where $E(w)$ is the embedding for word $w$.

The predicted output will be the tag with the highest probability, i.e.:

   $y_i = argmax_{j}P(t_{i,j}|w_{i-2}...w_{i+2})$

The output of the classifier should be a vector of size as the number of possible tags.
We can obtain this using function `to_categorical`, to turn tag indices into a one-hot representation

## Why classifier are not as good as sequence models

Common approaches in this case are probabilistic approaches, HMM, CRF [[Lesson 8 - Markov Random Fields#Conditional Random Fields]] see in ISPR course. And RNNs as well.

### Discriminative vs Generative

Discriminative compute directly the probability : CNN, CRF.

Discriminative are capable of inferring more information than

### MEMMs

In maximum entropy models the information flows differently, so $P(T | W) = \prod P(t_{i}| t_{i-i}, w_{i}, f_{i})$

We can still use Viterbi algorithm in MEMMs

# Named Entity Tagging

Tagging a document means to pick that and thing called *name entities* sequence of words that have certain meanings, for examples `American Airlines` are two word that stands for a company.
We have a lot of NE type for example, `people`, `organizations`, `location`, `facility` and so on.

### Biomedical entities and other difficulties

Like proteins with a lot of names: CREB2, ORF2b and so on. Or non contiguous overlap mentions:

"*Abdomen* is soft, *nontender*, nondistended, *negative bruits*"

What's the trick? Use more labels for annotate eg:

*Unified Airlines* said Friday it has increased ....

We give unified and airline B_ORG and I_ORG labels to mark beginning and end. And the other word are marked with O.

So we need for $n$ entities $2n+1$ labels (+1 because of the O syble).

## Using NER with Transformers

So the takeaway is to use a lot these NER in fact if one want to use a transformer with those have to start stack layers of those to get better results.


Some idea is when talking about LSTM : go in a direction using two LSTM one that goes left to right and other right to left, if there is something in context after long time, we could forget it, so we carry the information and blend them togheter to infer more information about the word of interests


>[!info]
> 






---
# References

