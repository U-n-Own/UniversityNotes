Date: [[2023-02-27]]

Status: #notes
Tags: [[Human Language Technologies]], [[Probability]], [[A.I. Master Degree @Unipi]].

[Slides](https://docs.google.com/presentation/d/16cUPpKlCXK9QKfgIzfkcZcLNX_dt8__2/edit#slide=id.p4)


Traditional way to build language model is to concatenate words and the words can be predictied with probabilistic models. So statistical methods are used a lot. Became impractical when you have a lot of words


### Probabilistic language model
The goal is to assign probability to a sentence, so we have the set of sentences in a language, some have no sense, are incorrect so probability of these is zero or very low. On correct and right phrases we have higher probability. For some sentences one can have more translation that are correct, but maybe some are more fluid, said more often, so that will be to one that has higher probability. An application where incorrect phrases are used it's mispell correction so those have lower probability. Also generating sentences: generative AI, can suggest something that more probably in context. Also speech recognition: you have a spoken sentence that maybe has different meaning but you use the most probable, the one that has more sense in a way...
With some fine tuning you get better model like LLMs tuned on question answering: ChatGPT.

### Mathematical definitions of Language models
The language model assign a probability $P$ to a sequence of words so $P(w_1, w_2...w_n) = P(W)$ is probability of a certain sentence, or we can use $P(w_5 | w_1, w_2, w_3, w_4)$ is the probability of the word $w_5$ given the previous 4 words. We can use the chain rule to get the probability of a sentence: $P(w_1, w_2...w_n) = P(w_1)P(w_2|w_1)...P(w_n|w_1...w_{n-1})$.
This is called **Language model**: model that computes $P(W)$.

### Chain Rule
We want to compute a join probability of a sentence that is made of words, so we know that

Definition of **conditional probability**:
$$
P(B|A) = \frac{P(A,B)}{P(A)}
$$
can be expressed as:
$$
P(A,B) = P(A)P(B|A)
$$
And in general:
$$
P(x_1,..x_n) = P(x_1)P(x_2|x_1)...P(x_n|x_1...x_{n-1})
$$

if we gave a sequence of words we get basically the chain rule applied to joint probability
$$
\prod_{k=1}^nP(w_k|w_1...w_{k-1})
$$

```ad-question
How we estimate P(the| it's water is so transparent that)?
```

But what's the limit? It will be very expensive to evaluate all words all possible prefix, more the word is big and a vocabulary is $500000$ words so for example for a $200$ long phrase we have $500000^{200}$ possible sentences. We will never be able to compute such statistics for long prefix. We need approximation

### Markov Assumption

The idea is that the model only remembers to a certain lenghts of past history and then old information become irrelevant, 1-order markov models consider only 1 previous words, computing only the probability of that. This assumption says that for each component in the product we replace an approximation like:

$$
P(w_k|w_1...w_{k-1}) \approx P(w_k|w_{k-1})
$$

##  N-grams 
We can use these to approximate with a $n$ that is small some text. We can extends the markov assumption to this model that is an approximation, this is insufficient models of language because of long dependances that are lost. eg "The *man* next to the large oak tree near the grocery store on the corner *is* tall" an error would be use *are* instead of *is* because these two are far away.

### Estimating Likelihood Estimate

$$
P(w_i | w_i-1) = \frac{count(w_i-1, w_i)}{count(w_i-1)}
$$
We just count how mant time a words occours after the previous one and then selection that one with more occurences.

![[Pasted image 20230227145154.png]]

So we need a probability distribution that "models" our data in a good way. We choose tha one that fits best our data. Suppose a words occurs $400/1000000$ times, so the MLE is $0.004$ .
Suppose we want to count how likely is to be infected by a virus, and we have a 10 cases, if $P$ is probability to be infected and $1-p$ is the opposite and we have 6 case out of 10 of infection, then $P(Data|p) = p^6(1-p)^4$ but this is very like a Bernoulli distribution. Taking the derivative of this function we get the MLE.

Set the derivative to $0$ we have: 

$$0 = \frac{d}{dp}p^6(1-p)^4$$
$$= p^5(1-p)^3 - (6 - 10p)$$

### Raw bigram counts 
![[Pasted image 20230227151827.png]]

We normalize by unigrams and we get:
![[Pasted image 20230227151910.png]]

### Issues
Compute in the log-space so compute the log of probabilities that is better because adding is faster than multiplying and we avoid underflow, because of multiplying small probabilities.

$$
log(p_1, p_2, p_3) = log(p_1) + log(p_2) + log(p_3)
$$

### Shannon game

1. Choose random bigram s, w according to it's probability
2. Choose a random bigram (w,x) according to it's probability
3. Repeat until you get s
4. String the words togheter


```note
This is actually used by chatgpt
```

### Approximating Shakepeare
![[Pasted image 20230227152636.png]]

More we go further more precise will become the model, and more you go more will sound Shakespear's like.
Shakespear produced 300.000 bigrams types out of $V^2 = 844$ million possible ones: 99,6% of possible will never seen entry in table are zero.

But be careful with the $N$ value, n-grams works really like if the text corpus looks like the training, in real life this doesn't happen, so be careful with overfitting, we need to train robust models.

### Smoothing
We have a problem with unseen bigrams, we can't compute the probability of a bigram that we never saw, so we need to smooth the model. We can use Laplace smoothing, we add a constant $k$ to the numerator and $kV$ to the denominator. This is called **additive smoothing** that is a *regularization* technique, instead of letting values go to zero, we want them to be small, if an entire sequence gets $P(W) = 0$  we get infinite perplexity measure. We can also use **interpolation** that is a weighted average of the unsmoothed and smoothed models. We can also use **backoff** that is a weighted average of the smoothed model and the model of the previous order.

### Smoothing is like Robin Hood
![[Pasted image 20230227153423.png]]

As we said before the best way to do this is the **Laplace estimate**, will just add $1$ to the count.

$$
P_{Laplace}(w_i) = \frac{c_i + 1}{N+V} 
$$

Also big changes to coutns have effects, other way are to add small fractions of 1.

#### Code example: Generating 20 bigrams from "my"
![[Pasted image 20230228091830.png]]

### Zeros or not?

```ad-important
Zipfs-Law: a *small* numeber of events occur with *high frequency* and large number of events occurs at *low frequency*
```

Results are that the estimates are **sparse** 
![[Pasted image 20230228092717.png]]

There is a correlation to frequence and rank, this law tells us that the $f \approx 1/r$.

Here the distribution in $log$ space
![[Pasted image 20230228093009.png]]

##### Interpretation of Zipf's law
Principle of **least effort** both speaker and hearer in communication want to minimize efforts

- Shrink vocabulary -> speaker
- Less ambiguous vocabulary -> hearer

Zipf's law is a compromise between the two

### Coding zipfs law

Implementation works but not here

```run-python
import nltk
import matplotlib.pyplot as plt
from collections import Counter

nltk.download('brown')
from nltk.corpus import brown

tokens = brown.words()

token_counts = Counter(tokens)
sorted_tokens = token_counts.most_common()



ranks = list(range(1, len(sorted_tokens)+1))
frequencies = [count for token, count in sorted_tokens]

plt.scatter(ranks, frequencies)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title("Zipf's law in the Brown Corpus")
plt.show()

```

```ad-question


```

## Evaluation

- Intristic evaluation: Use the model to evaluate itself, for example with **perplexity**, this one approximate poorly the performances.
- Estrinsic evaluation: use a task to evaluate the model, compare two models and choose the best one, this one is more complicated and expensive time-wise.

### Perplexity
Perplexity is a measure, intrisic type evaluation and the basic idea is how much surprised the model is when we show the test set. The more suprised is the model much lower will be the probability of the test set, the other way around for less surpise.

Mathematically speaking
$$
PP(w) = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i|w_1...w_{i-1})}}
$$

Using the chain rule we can get the perplexity. What we want is to minimize the perplexity, so maximizing the probability. Lower perplexity better the model.
What we expect is that more vocabulary there are, higher will be perplexity.

What if we have words unknown in the test that we don't had in training set? Common solution is to introduce a special token: <UNK>, so we choose a vocabulary in advance and replace words that aren't there with the token and then we train.


- - -

# References

Code at lesson [here](https://medialab.di.unipi.it:8000/user/v.gargano1@studenti.unipi.it/lab/tree/HLT/Lectures/LanguageModel.ipynb)

[Google N-grams](http://ngrams.googlelabs.com/)