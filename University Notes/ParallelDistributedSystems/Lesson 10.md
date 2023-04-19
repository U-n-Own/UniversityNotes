Date: @today

Status: #notes

Tags: #spm

# Title

```ad-important
Assignment 4: Stencil code with OpenMP

Pick a $x\times n$ matrix of floats A, where $A_{ij}^l = avg(neighborhood(A_{ij}^{l-1}))$

Where a neighborhood is diagram is: use markdown to give an example# Neighborhood Diagram Example

| Column 1 | Column 2 | Column 3 |
|---------|---------|---------|
|     |         |         |
|         | *       |         |
	| *    |    $A_{ij}$     | *  |
|         | *  |         |

This is an example of a neighborhood diagram.

We for stencil code want to Vectorize

$$
\sum_{ij} |A_{ij}^{i+1} - A_{ij}^i|
$$

As in all stencil computation we have to keep value until we don't need them for any of the next computation

For example at the start of the matrix we will utilize the item in the row above when we get to compute the values, we need a buffer for the items at previous step and another for the current row
```
```c++
for (rows)
	for(cols)
	buff[][]
	a[][]
	current_buff[][]
```

```ad-important
A faster way is to use double buffering, using A,B

```
```c++
while(){
for( ){
	for( ){
	b[i][j] = f(a[i][j]...);
	}
}
swap(a,b); \\Or std::swap(,) This uses double amount of memory
}
```

## Dependancies and conditional execution

Depending if i'm in the border i've to consider 3 and depending if we're at the angles of the matrix we've to dheck different problem

So we can use 

`#pragma omp for`
`#pragma omp taskloop`
`#pragma omp tasks`

We need to decide where to use parallelism


## Second part of lesson: GRPPI (EC3M)

Patterns

### Take home lesson

```ad-summary


```


---
# References

