
```json
---
share: true
---
```

[[Clustering]], [[Data Analysis]],[[Expectation Maximization]],[[Probability]], [[Data Mining]]

Clustering algorithms require as a parameter the number of cluster we could imagine to see into our data, then the algorithm optimize function that taking this into account can

## Why **Hierarchical clustering**?

When the structure in the data is distributed in pyramidal fashion, like products in a supermarket, there are those that are most bought or cost more and so on. **Dendrogram** that can be represented by sets (overlapping elements between two sets or more).

## Why Partitional Clustering

Well this tends to have a well separated cluster.

## Fuzzy clustering

We can introduce probability that a certain datapoint to be associated to a certain cluter, so the algorithm will work using a statistical model and will look at the probability to be in a certain cluster

# K-means 

Most famous algorithm for clustering.

Idea because K-means sucks: Apply K-means with a lot of cluster and then group the clusters.

```pseudo
\begin{algorithm}
\caption{K-means}
\begin{algorithmic}
  \Procedure{Kmeans}{$X, K, \epsilon$}
    \State Initialize $K$ cluster centroids randomly
    \Repeat
      \For{each point $x$ in $X$}
        \State Assign $x$ to the nearest centroid
      \EndFor
      \For{each centroid $c$}
        \State Update $c$ as the mean of points assigned to it
      \EndFor
    \Until{convergence criterion is met (e.g., small change in centroids)}
  \EndProcedure
\end{algorithmic}
\end{algorithm}
```

### K-means++ variation

We could do the update incrementally, but the probability to be picken is defined as $d(x)^2$ as the *squared distance to the nearest centroid*

Since the outliers are quite rare we don't have to worry

```pseudo
\begin{algorithm}
\caption{K-means++ Initialization}
\begin{algorithmic}
  \Procedure{KmeansPlusPlusInit}{$X, K$}
    \State \textbf{Step 1:} For the first centroid, pick one of the points at random.
    \For{$i \gets 1$ \To $\text{number of trials}$}
      \State \textbf{Step 2:} Compute the distance, $d(x)$, of each point to its closest centroid.
      \State \textbf{Step 3:} Assign each point a probability proportional to each point’s $d(x)^2$.
      \State \textbf{Step 4:} Pick new centroid from the remaining points using the weighted probabilities.
    \EndFor
  \EndProcedure
\end{algorithmic}
\end{algorithm}
```

### Bisecting K-means

Assume entire dataset is a cluster put all into a list, and then work to find two cluster that has less SSA, this will produce **Hierarchical Clustering** structure like a binary tree

How do we decide the next cluster we want to split? Bigger SSA, Bigger Cluster or both maybe weighted...

```pseudo
\begin{algorithm}
\caption{Bisecting K-means}
\begin{algorithmic}
  \Procedure{BisectingKmeans}{$X, K, \epsilon$}
    \State Initialize a single cluster with all data points
    \While{number of clusters < K}
      \State Select the cluster with the highest variance or the larger or with highet SSA
      \State Apply K-means to split the selected cluster into two sub-clusters
      \State Remove the selected cluster
      \State Add the two sub-clusters to the set of clusters
    \EndWhile
    \State Apply K-means to the final set of clusters until convergence
  \EndProcedure
\end{algorithmic}
\end{algorithm}
```

## Limitations of K-means

Non globular shapes can give problems, as we said solution is to do some postprocessing by splitting the clusters, how to merge? Take two centroids and take their distance and merge them, then continue until the desired number of centroids is reached by merging.

Is useful to compute the distribution of the SSA because sometime we can get empty clusters (extreme case), but other times we have a small cluster very close with few element near a big cluster, so we would like to merge them (the nearest cluster) so analyze the SSA if we merge them and see what happens.

### Pre and Post processing

Normalization and outliers elimination is preprocess

Most of post processing is merging togheter cluster, get rid of bad clusters.

### Poor initialization and some solutions

**Problem**:

Well what do we do since if the initialization (random) is bad then we have bad cluster?

**Solution**:

One can be select a random point then for the next cluster choose the farthest point from the points already selected as centroids: yeah and then we select outliers and ruin our clustering, so what to do? Do it more times...

Multiple runs is not always feasible because of the amount of data we have sometimes, also is expensive the finding the furthest point if the points are a lot and what we do? [[Sampling]]!

#### Probability to get right clusters

Let's say that we want to know 
$$P= \frac{\text{number of way to select one centroid from each cluster}}{\text{numer of ways to select K centroids}}$$
## X-means

Yet another extension of K-means which try to determine number of cluster based on a score: **BIC score**.

The origin of X-means is the random initialization, can we do better?

1. K-means is slow and scale poorly on big sets of data
2. The number of cluster has to be predetermined beforehand
3. When it runs with fixed values of K, using empirical rules it finds local optima instead of a dynamical approach

So *X-means* comes into play after a K-means run, making local decision about which subset of the current centroids split to get a better fit, how do we decide? We introduced it: BIC!

What is BIC?

```ad-example

Bayesian Information Criteria: 

$$
BIC = \ln(n)k-2\ln({\hat{L}})
$$

where $\hat{L}$ is maximize value of likelihood of the model aka [[Lesson 5 - Generative and Graphical Models#Bayes Rule (ML interpretation)]](How much probable is to get those data **given the parameters**), $n$ is number of datapoints and $k$ number to estimate parameter

Is basically a tool for model selection: BIC score gives an estimate of the model performance on a new, fresh data set (testing set).
BIC attempts to mitigate the risk of over-fitting by introducing the penalty term $k\ln(n)$, which grows with the number of parameters. This allows to filter out unnecessarily complicated models, which have too many parameters to be estimated accurately on a given data set of size N. 

```
 
So at the end lower the BIC better the model. 
## Clustering with Mixture Models

Now we can see a way to cluster that is based directly on the data distribution, so we use a number of distribution to model our data.

Usually when we talk about data distribution we have those parameterized, and we need to find the right parameter. Now enters into play **MLE** : [[Lesson 5 - Generative and Graphical Models#Differentiating MAP, Bayesian and ML]] and [[Lesson 7 - Hidden Markov Model#MLE learning parameters!]]

So we can use EM-algorithm to cluster data, basically we give a initial guess and then iteratively refining


So mixture of what? Usually we take multivariate normal distributions, we want to model elliptical cluster structures, usually mixture model have to objetive to generate data, if you have those distribution take one at random (each has different parameters) and sample from it. Repeat this $m$ times 


Let's start simple by using a simple Univariate Gaussian Mixture

$$
p(x|\Theta) = \frac{1}{\sqrt{2πσ}}e^{−\frac{(x−μ)^2}{( 2σ^2)}} .
$$

So our parameters in this case are ($\mu,\sigma$) = $\theta$.

![[Pasted image 20231012151616.png]]

Now in order to get the best set of parameter that fits our data we will use MLE.

Given a sample of points as we can see we want to find the distribution $\theta$ parameters that generated them *most probably*, so what we need is the [[Likelyhood]]:

$$
\begin{align}
	LogLikelihood(\theta | \mathcal{X}) = -\sum_{i=0}^n \frac{(x_i-\mu)^2}{2\sigma^2}-(0.5n)\log2\pi-n\log{\sigma}
\end{align}
$$
We optimize with respect of log likelihood since it's monotonically increasing and we can reduce the product to a sum...

## EM for estimation

Ok now we move toward a more difficult topic, what if data is generated by a mixture of gaussians? Well we got more parameters...
Plus we now have points, but we cannot assume there're all from the same distribution, we cannot multiply their probabilities!

We will leverage on EM, that will give the probability of each point being generated by each distributions then use this to a better estimate.  


```pseudo
\begin{algorithm}
\caption{EM Algorithm}
\begin{algorithmic}
\State \textbf{Initialization}:
    \State Select an initial set of model parameters. (Randomly or in various ways)
\Repeat
    \State \textbf{Expectation Step}:
        \For {each object $x_i$}
            \State Calculate the probability that it belongs to each distribution
            \State Calculate $\text{prob}(j|x_i, \Theta)$ for each object
        \EndFor
    \State \textbf{Maximization Step}:
        \State Given the probabilities from the expectation step, find new estimates of the parameters that maximize the expected likelihood.
    \State \textbf{Convergence Check}:
        \State Check if the parameters have stopped changing significantly.
        \State Alternatively, stop if the change in the parameters is below a specified threshold.
\Until{Convergence}
\end{algorithmic}
\end{algorithm}
```

Intrestingly K-means for euclidean data is a special case of EM algorithm for gaussian distribution, where expectation step corresponds to K-means assigning each object to a cluster and maximization to compute the cluster centroids.

## Why and Why not using Mixture models against simple K-Means?

Mainly negative side of EM is the fact that is slow, it doesn't work well for linearly correlated data and if there are too many points is unpractical, but the thing is that these methods are more general than K-means, usually these methods like c-means and clustering are forcing the data to be captured in some distribution, on the other hand Mixture start from a very general shape (elliptical) trying to characterize the exact distribution.