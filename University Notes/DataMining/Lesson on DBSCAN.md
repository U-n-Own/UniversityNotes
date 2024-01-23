[[DBSCAN]],[[Clustering]],[[Data Mining]]

# Yet another Density-based algorithm

Let's start by giving some terms to speak the same language in order to understand this.

```ad-abstract
title: Definition
collapse: open

Density: Number of points inside a specified radious $\epsilon$

```

A point is a *core point* if has a certain number of points (MinPts) inside $\epsilon$, so points internal to cluster and the point selected is one of them.

A *border point* is **not** a core point but is in it's neighborhood

A *noise point* is any point that is not a core nor border point


![[Pasted image 20231019181552.png]]

## DBSCAN algorithm

```pseudo
	\begin{algorithm}
	\caption{DBSCAN algorithm[MinPts, $\epsilon$]}
	\begin{algorithmic}
		  \State \textbf{Step 1}: Label all point as core, border or noise
		  \State \textbf{Step 2}: Eliminate noise points
		  \For{all core points}
		  \State \textbf{Step 3}: Create an edge between core points with a distance $\leq$ $\epsilon$
		  \EndFor
	\State \textbf{Step 4}:Make each group of connected core points separated into cluster
	\State \textbf{Step 5}:Assign each border point to the nearest cluster of it's associated core points
	\end{algorithmic}
	\end{algorithm}
```

### DBSCAN parameters

As for the parameters MinPts and $\epsilon$ we have to give them beforehand, how do we do?

Basic approach is to look at the behaviour of a point to it's $k-th$ nearest-neighbour let's call it: **K-dist**; in average if the point is in a cluster and $k$ is less than the number of points in the cluster this value will be small.

For noise points this value will be large relatively to the others, *we compute this values for all datapoints for certain $k$ and then we put them in an ordered list*, and then we plot. As soon we see a sharp change we can clearly say that that jump corresponds to our $\epsilon$ and the $k$ will be our MinPts.

We can see the ordered list below

![[Pasted image 20231019184510.png]]


## DBSCAN problems & advantages

What is cluster density vary with high variance between them? Then we could fail to identify some clusters due to the measure of the *k-dist*. So high density is clusterized and low density is marked as noise.

Troubles comes also for high dimensional data, like the [[Curse of Dimensionality]] in ML the density of the points is completely destoryed

Being density based can be also useful to eliminate noise and is capable to identify strangely shaped cluster that sometime we can have in our data. 