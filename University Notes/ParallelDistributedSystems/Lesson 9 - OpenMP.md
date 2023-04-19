Date: @today

Status: #notes

Tags: [[Parallel Computing]]


# Pragma how they works

when we write

```c++
#pragma omp parallel{
...
}
```

All the code inside is run using the most resources than we can do, we can also decide if we want to use a certain number of threads using 

```c++
#pragma omp parallel num_threads(num)
```

In bash we can do 

```bash
export OMP_NUM_THREADS=6
.\a.out
```

Another sets of pragma we can use are  those single, atomic and critical

- Single pragma make execute a single thread the code
- Critical section define a critical section
- Atomic make all the code execute like a critical section

### Vectorizing

```c++
#pragma omp simd
```

Tomorrow three work sharing pragmas: for, task, taskloop

Vectorization and OpenMP are good in the book chapter 6,7

### Take home lesson

```ad-summary


```


---
# References

