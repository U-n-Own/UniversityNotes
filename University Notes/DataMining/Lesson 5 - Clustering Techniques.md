[[Data Mining]], [[Clustering]]

## [[K-means]]

Partitional clustering approach, we need to specify a number of cluster $K$, each cluster is associated with a centroid and each point is assigned to the closest centroid hence a cluster

![[Pasted image 20230704193200.png]]

Often we choose centroids randomly and then the next centroid is the average of the points in the cluster.
Closeness is measured usually with euclidean distance

Complexity is $O(nKId)$

Where $n=$ number of points, $K$ number of clusters, $I$ iteration and $d$ number of attributes

## K-means evaluation

![[Pasted image 20230704193441.png]]

## Big problem: Initial centroid problem

If we assing as initial centroid the wrong points we could get a suboptimal clustering, so what can we do? Retrying is expensive and probability wise if the space is big then we have low probability to gess the right initialization of centroids, more solution are using sampling to reduce space, postprocess data use larger number of cluster and use hierarchical clustering or select more than k initial centroids and then take the one with the best SSE.

## Convergence of K-means

![[Pasted image 20230704194453.png]]