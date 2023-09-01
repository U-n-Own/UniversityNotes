[[Data Mining]]

# What is cluster analysis

```ad-important
Finding a group of items such that all items in the group are similar between them and different from other group by specifiying some [[Lesson 3 - Data Similarity]] measure.
 ```

![[Pasted image 20230704161917.png]]

## What is not cluster analysis?

- Dividing items by some order: Segmentation
- Querying is not clustering
- Classification is not clustering (supervised).
- Association analysis

## Types of clustering

**Partitional clustering** is the intuitive clustering we spoke about where we classify in our cluster items based on some similarity metric

**Hierarchical clustering** instead is the one where the cluster can have "subclusters" and we can find a tree structure called *dendrogram*


![[Pasted image 20230704163256.png]]

More compelx dendrogram, we can use a cut to determine what clusters we need "at what level of hierarchy"

![[Pasted image 20230704163556.png]]

## Type of clusters

```ad-info
A cluster is a set of points such that any point in a cluster is  
closer (or more similar) to every other point in the cluster than  
to any point not in the cluster.
```

### Well separated cluster

All points are in a cluster and each cluster doesn't overlap

### Center based

The center of a cluster is called **centroid** is the average of all the points in the cluster

### Contiguity based

Each point is closed to at least one point in it's cluster than any point in other cluster : Nearest neighbour 

![[Pasted image 20230704164033.png]]


### Density based

A cluster is a dense region of points separated by less dense regions from other high density regions

## Defyning clusters

We can visualize clusters by an objective function, so we want to find the clustering that minimize or maximize our objective function.