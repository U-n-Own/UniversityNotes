[[Data Mining]] [[A.I. Master Degree @Unipi]]

## Sampling reduce dimensionality

[[Lesson 10 - Sampling Methods]] we've seen in other course can be used for reduce dimensionality of data, how?

Well **Sampling** in context of data mining is used when we have too much data in our datasets, so we reduce dimensionality by sampling from our dataset a representative collection of points.

- Random Sampling with or without replacement
- Stratified Sampling: Split data in partitions and then draw from each class 

## Curse of Dimensionality again

![[Pasted image 20230703162508.png]]


## Reduction methods

- Filter methods: Pre processing by analyzing the attributes
- Wrapper methods: Selecting top ranked features using some statistical significance, eg Forward selection by adding feature that improve the best performance or remove one by one the ones that are less signficant
- Embedded methods: Using some DM algorithms like decision trees that will drop attributes.


- Feature creating: Use a unique measure to embed in it 1 or more other features that are row data used to compute a measure like efficiency

![[Pasted image 20230703163732.png]]

- Feature projection

Techniques used to further reduce dimensionality are [[Principal Component Analysis]] and [[Singular Value Decomposition]] we've seen in [[Computational Mathematics]] course.

Going from high dimension to lower dimensions

### Missing values and Data Cleaning

Instead of missing values use *mean*, *mode* or *median*.

## Discretization

Going from a space where values are continues to a more small space where are discrete, for example take the height we can divide it in classes: small $[1,20-1,50]$ medium $[1,51 - 1,75]$ tall $[1,76 - 2]$.

We want some *binning* techniques: Natural binning, frequency binning and statistical binning.

### Natural Binning

Unusual values go in bin with small quantities, so we can detect outliers

![[Pasted image 20230703170514.png]]

### Frequency Binning 

All bins contais about the same number of samples

![[Pasted image 20230703170603.png]]


## Entropy based approaches

Minimize the [[Entropy]] wrt a label or maximizing **Purity** of an interval what is the purity?

![[Pasted image 20230703171357.png]]

We use confusion matrix to calculate the purity.

A simple approach is this, bisecting initial values so that the intervals give minimum entropy then we keep splitting with another that has higher entropy and so on.

![[Pasted image 20230703171553.png]]


## Data Transformations

Why transform data?

If they are incomplete, not adeguately distributed (asymmetry or peaks)

### Attributes Transformations

We take some function that map set of values or attributes to another space like linear basis 

Main goals of these are: preserve the original assumption on our data and stabilize variance, normalize our distributions, linerize relationships in our data. Also represent in a different scale our data for example in log scale.


### Normalization

![[Pasted image 20230703172440.png]]

### Transformations

For non linear relationships or skewed distribution

Before transformation: $10,000, $20,000, $30,000, $40,000, $100,000, $200,000, $500,000, $1,000,000

After logarithmic transformation: 9.21, 9.90, 10.31, 10.59, 11.51, 12.21, 13.12, 13.81

![[Pasted image 20230703172801.png]]

## PCA 

![[Pasted image 20230703172838.png]]

1. Standardize data
2. Calcualte *Covariance Matrix*
3. Calculate the $\lambda$ and $\lambda \bar v$ of the Cov Matrix
4. Sort $\lambda$ and $\lambda \bar v$ and take take $k$ eigenvalues and form a matrix of eigenvectors
5. Transform the original matrix

*Variance* and *Covariance* are measure of how **spread** is a distribution from our mean and a measure of how each dimension vary with respect to the others, by taking them as couples. 

Eg Covariance between hour of study and grades taken.

The covariance between a dimension and itself is the variance!

![[Pasted image 20230703173455.png]]
![[Pasted image 20230703173647.png]]