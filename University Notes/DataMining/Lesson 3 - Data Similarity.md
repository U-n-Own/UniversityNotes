
[[Data Mining]], [[A.I. Master Degree @Unipi]]

# Similarity and Dissimilarity

Estimating how two datapoints are similar in some way. We have some tools to detect similarity like cosine similarity, (dot product). Often a measure of similarity fell in [0,1] and it's higher when they are similar.

Dissimilarity is like the reverse: lower for similar objects

![[Pasted image 20230703175651.png]]

## Similarity as distances

### Euclidean distance

![[Pasted image 20230704145027.png]]


### Minkowski Distance

![[Pasted image 20230704145126.png]]

#### Properties of distances

![[Pasted image 20230704145219.png]]

### Binary distances 

What if our objecta has only binary attributes?

Well we need a similarity between binary vectors

![[Pasted image 20230704145608.png]]


### Cosine similarity for document data

![[Pasted image 20230704145807.png]]

### Weighting similarity

We can give more importance to certain aspects or attributes by assigning weights 

![[Pasted image 20230704145855.png]]

## [[Correlation]]

![[Pasted image 20230704145934.png]]

### Visualizing correlation

![[Pasted image 20230704145951.png]]

## [[Entropy]]

Entropy measure of uncertainty or "surprise", this measure quantify how much information we gain by knowing the outcome of a certain event, we define a bit of information as the measure of entropy, when we gain one bit of entropy we divide the space of the possible guesses of half.

![[Pasted image 20230704150751.png]]

### Entropy for samples

![[Pasted image 20230704150900.png]]

## Mutual [[Information]]

Information of a RV usually is defined as

$$
I(X) = -\log_2(P(x))
$$

If the outcome has probability of 1, we have 0 information content since it carries no uncertainty, would be infinite in case of 0 probability

Instead mutual information is the one that we get from two variables in a joint distribution, easy for discrete variables, but not that easy to compute for continuous one.

![[Pasted image 20230704150947.png]]


### Mutual info example

![[Pasted image 20230704151514.png]]