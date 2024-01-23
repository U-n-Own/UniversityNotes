
```json
---
share: true
---
```

[[Clustering]], [[Data Analysis]], [[Data Mining]]

# Hierachical Clustering

In the previous lesson on the introduction of clustering we talked a little about it [[Lesson on Clustering (Centroid based)#Why **Hierarchical clustering**?]]

We got two practical approaches to this:

1. Agglomerative: Start with points as individual cluster and then find the points that are more similar and merge them, needs a right definition of proximity measure.
2. Divisive: Start with a big cluster and then split until getting singletons of points or with a certain number of cluster $k$ predefined

The two techniques are quite the opposite one of another.

## Agglomerative Clustering

```pseudo
	\begin{algorithm}
	\caption{Agglomerative Clustering}
	\begin{algorithmic}
		  
    \State \textbf{Step 1:} Compute the proximity matrix.
      \State \textbf{Step 2:} For each point to be a cluster
      \Repeat
	      \State \textbf{Step 3:} Merge to closest clusters
	      \State \textbf{Step 4:} Update proximity matrix 
	 \Until{Only 1 or $k$ clusters}
	\end{algorithmic}
	\end{algorithm}
```

How do we evaluate proximity of two clusters? We've seen a lot of ways: Use only the centroid, use all the points inside and maybe mean it, use the min or max distance
### Min Metric vs Max Metrix

These first two measure within, the *Group Average* (proximity of two cluster is the proximity of the pairwise points inside them) are coming from graph theory.

![[Pasted image 20231019171612.png]]

This is quite good for these shapes, but sensitive to outliers and noise.

![[Pasted image 20231019170909.png]]

For the max, we can imagine that is penalized if points are too near each other, but it's useful if the points are in a globular fashion.

![[Pasted image 20231019171039.png]]

### Ward's Method: a prototype base view

[[Ward's method]] assumes that a cluster is represented by it's centroid but measure the proximity by the SSE change in the merging of clusters, like K-means it want to minimize the sum of squared distances of point from their centroids.


```ad-note
We can see that in some specific way, this method is basically a *Group Average* if the proximity on the group average is defined ad sum of square of their distances.
```

![[Pasted image 20231019172621.png]]

#### A problem with centroid based 

As we said this is the analog of centroid based in heirachical fashion, and one problem is the possibity to have **Inversions**, that are: 

```ad-attention
Two cluster merged at step t can be more similar (less distant) than pair of clusters merged in previous steps t-1 for these hirearchical methods the distance in monotonically increasing!
```

## Divisive Clustering

```pseudo
	\begin{algorithm}
	\caption{Divisive Clustering (MST)}
	\begin{algorithmic}
		  
    \State \textbf{Step 1:} Compute a Minimum Spanning Tree.
      \Repeat
	      \State \textbf{Step 2:} New clusters are created by eliminating an edge, deleting the largest distance
	 \Until{Only 1 or $k$ clusters}
	\end{algorithmic}
	\end{algorithm}
```

...more in the book
## Time and Space Complexity

Take $m$ in mind as *number of datapoints*

This is quite intresting, because we require to store a proximity matrix, and we can use less if the matrix is symmetrical but storing only the triangle matrix but still quadratic, the intresting part is how we can simplify the time complexity by leveraging the structure being ordered! 

Being still quadratic in order to compute the proximity matrix we have to be careful to Step 3 and 4, if we do it linearly searching we have to merge with $\mathcal{O}((m-i+1)^2)$, and then update $\mathcal{O}((m-i+1))$, yielding a cubic complexity.

Update(quadratic)+search(linear) nested, we can do better by ordering, so the search is constant but we pay only the order step, less

Space  Complexity: $\mathcal{O}(m^2)$
Time Complexity:  $\mathcal{O}(m^2\log(m))$


## Hierachical clustering problems and limitations

- Once you combine clusters there is no way to go back.
- No direct function is minimized
- There are problem with outliers, non globular shapes and different cluster sizes also if cluster are too large