Date: [[2023-02-28]], [[2023-03-01]]

Status: #notes

Tags: [[Human Language Technologies]]

# Words

By definition the meaning is:
1. The idea that is represented by a word, phrase 
2. The idea that somewant want to express with that word
3. Or maybe some other hidden meaning that is expressed via arts

Purely speaking we have **signifiers** and **symbles**.

## Linguistic Solution

We have some dictionary and lexical resouces like [Wordnet](https://wordnet.princeton.edu/). Containing  synonymous

Check the Wordnet jupyter on the moodle....

Concept of synset: a set of synonymous words

Higly context dependant so we have a certain limit, another drawback since language evolves and words assume new definitions, so we need to update the dictionary.

### Vector Space Model
VSM is a representation for text used for Information Retreival. A document is represented by a vector in n-dimensional space $v(d_1) = [t_1, t_2...,t_n]$ each dimension is a separate term and then you can take these vectors and measure the similarity, computing the cosine between vectors, the larger the more similar.

eg. motel, hotel, city

Vector representation for hotel: $[0 0 0 0 0 0 0 1 0 0 0]$
Vector representation for motel: $[0 0 0 0 1 0 0 0 0 0 0]$

%place image here

A document that contains occurences of a word like "game" that value of weight will grow $df_i$ document frequency of word $i$

$$
idf_i = \log\frac{N}{df_i}
$$

#### Sparse Representation

Salton et al suggeste the $tf * idf$ weighting to represent docs. But still the representation is very sparse and semantic similarity is not captured.

#### Discrete symbols 
All vectors are orthogonal, so we can't capture the semantic similarity. Search engines address this issue using synonyms, but this leads to more problems like the polysemy problem.

```hypotesis
Co-occurring words are semantically related
```

Some sentences make us undestrand that some words have different meaning depending on the context, we can exploit this property following the **distributional hypothesis**.

#### Intuition: Embeddings
We embed the meaning in a vectors space, the meaning of a words will be a vector so word vetors are **Word Embeddings**.
Now we have four kinds of vector representation:

1. Mutual information weighted
2. SVD
3. Neural-Networks inspired models
4. Brown Clusters

The idea of Distributational Hypoytesis is an old idea and is basically this

```ad-quote
You shall know a words by what is near that word, the context
```

#### Context
A context is a matrix $|V| \times |V|$ this matrix make us see that words and context words are very correlated, they appear togheter in phrases with some frequency, rows of $X$ capture similarity, but this matrix is still sparse, $X$ is the Co-occurrence matrix. But neighboring words are not semantically related

%image here 

## Word Embeddings
Project words vectors $v(t)$ into low dimensional spaces, of continuous word representation (aka) **embeddings**.

$$
Embed: \mathbb{R}^{|V|} \rightarrow \mathbb{R}^k
$$
$$
Embed(v(t)) = v(t)
$$

The most important paper is from Collobert et all (2011), Natural Lnguage Processing (Almost) from Scratch, they use a neural network to learn the embeddings. 
Using this Loss called **Hinge Loss**:
$$
\mathbf{L}(\theta) = \sum_{x \in X} \sum_{w \in V} max(0, 1 - f_\theta(x) + f_\theta(x^{w}))
$$

% insert image

The innovation is that they used as input positive samples as word vectors and negativa samples are obtained by changing one word with another from the dataset, usually the other samples are so many that is almost impossible to get a good result in the negativa one.
Now with words embeddings we have neightbors semantically related.

### Word2Vec
In 2013 Mikolov et al. proposed a new model called Word2Vec, this model is based on the idea of **skip-gram** and **continuous bag of words**. This neural networks estimates probabilities from a large corpus of text and tends to maximize the probability to predict a context words.

This words embedding is unsupervised, so no need human labeled data. You simply split words in tokens, but actually there are supervision because the sentences used are written by humans.
Also size of words embeddings matters.

#### Skip-gram 
Predict context words withing a window. Given a sentence we turn that into an embedding, one could take $V$ word embedding matrix and multiply it by one-hot encode vector word the word, we get a vector $v(t)$, and then we do trough a dense fully connected layer that gives us a $z$, then $softmax(z)$ and we get the probability distribution of the words in the context.

%skipgram image

Remember that some words probability will be zero, but that would be a mistake.

#### How Softmax works

*Softmax* convert $z$ into a probability distribution $p_i$. With $v_t = w_tV$ and $z = v_tU$ we have:

$$
p_i = \frac{e^{z_i}}{\sum_{j\in V}^{|V|} e^{z_j}}
$$

The denominator is the normalization factor, so the sum of the probabilities is 1.

#### Procedure

- Take a document split it in windows
- For each word in the window, take the context words
- Generate a score vector $z$
- Turn the $z$ into a probability distribution $p$ with *softmax*

With $\hat{y_{c-m}}, ..., \hat{y_{c+m}}$ are estimates of probabilities.

#### Objective function

For each pair of words we compute the **Likelyhood**

$$
Likelihood = L(\theta) = \prod_{i=1}^{T}\prod_{-m \leq j \leq m, j \neq 0}^{} P(w_{t+j}|w_t)
$$

Then we take the loss $J(\theta)$ as negative likelyhood.

$$
J(\theta) = -\frac{1}{T}\log L(\theta)
$$

Using $\log$ we can do some semplification by using the **chain rule** and summing.

Softmax function maps arbitrary values of $x_i$ 

## Word: What's a concept?

Thisi s rather a philosophical question, debated since Aristotele and Plato then late Kant, Frege. Chomsky.
Another question is: can we have concept **without language**? Sometime the concept that we extract from language is different from the true one...

```ad-quote
Some body 
```

### Training Skip-gram 

**Naive Bayes Assumption** context words are independent so we can multiply the probabilities.

$$
\prod_{j = -m, j \neq 0}^m P(w_{t-j} | w_j)
$$

This is similar to how children learn language, speaking to children make them able to absorb language, learn by hearing and then try to speak themselfs.

The $T$ in the formula above is the number of fragments of sentences in the corpus.
Lets calll $P(o | c)$ the probability of a word $o$ given a context word $c$.
The gradients:
$$
\frac{\partial J(\theta)}{\partial v_c} \;\; \frac{\partial J(\theta)}{\partial U}
$$

### Model parameters 
$\theta$ are our model parameters in our case with $d$ dimensional vectors and $N$ words the theta is a vector.

To compute now the gradient we do good old gradient descent.

### Cross Entropy Loss
The cross entropy loss is a loss function that is used in classification problems. It is defined as the average negative log-likelihood of the true class.

```python
def cross_entropy(X, y):
	"""
	X output of FCL
	y are labels of correct class
	"""
    m = y.shape[0] # num of examples
    p = softmax(X)
    # p[range(m), y] slices estimates for correct classes
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss
```

Words embedding were too difficult to compute there are some tricks to speed up 

- Subsampling: vocabulary reduction, removing unfrequent words eg. less than 5 occurrences
Words discarded with probability $P(w_{i)}= 1-\sqrt t$
- Parallelize: use multiple cores, since computation is stochastic we don't mind and one can do quickly.
- Negative sampling.

### GloVe: global vectors for word representation
The ration of conditional probabilities may capture meaning.

## Similarity of word vectors

One way to compute it to do the cosine between the two vectors, another way can be use $L_2$ norm. Cosine is often used in information retrieval, the cosine tells you how far the vector are each other. These distances are the start of words embedding building, also the example saw at lesson is a very small one, today's model are

## Using PCA 

One can as well use some PCA on the large corpus of vectors, and made it spit out in a lower space from 100 dimensional space to 2 dimensional space

![[Pasted image 20230306153748.png]]

By doing this on a co-occurrence, one obtains a thing very similar to word embeddings, Levy and Goldberg proved that skipgram models factorizes a PMI (Pointwise Mutual Information), what is important is the number of parameters and data used to learn the model.

![[Pasted image 20230306151524.png]]

### Visualization with t-SNE

Useful for nigh dimensional data preserving distances, with PCA we tend to lose that informations. Also embeddings are very bad at capture insightful meaning between words like this.
![[Pasted image 20230306152117.png]]

### Evaluation 

Some embedding are pre computed, also the way usually one embedding is showed to be good is to semantic properties between words, so **analogy evaluation** that is a form of Intrinsic evaluation.

## Embedding in Neural Nets

An embedding is used in the first layer of a classifier, we train the embedding then we feed these one such that we don't start with a random initialization, using learned embeddings speed up a lot the training, this was NLP for a lot of years. 

## ELMo 

A new breed of language models: Embedding from Language Model, we use before a NN to learn the embeddings, instead here we use a sequence of $n$ tokens the model predict the *probability* of the next token given the history. 

## Bert : Bidirectional Econder rerpesentation from Transformers

The task that uses is to predict random words from a sentence. Taking random words from a text and try to predict the next word given those. Bert uses a positional embedding to parallelize the computation.

## OpenAI : GPT-2 (2019)

GPT-2 Is a decoder for generating sentences

## GPT-3 

A larger model from GPT-2 with 175 billion parameters



>[!info]
> 






---
# References

Implementation of Sentiment specific word embeddings: DeepNL: https://github.com/attardi/deepnl

