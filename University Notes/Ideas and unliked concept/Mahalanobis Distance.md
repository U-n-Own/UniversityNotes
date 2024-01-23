[[Data Mining]]

Taken two variables they can have a correlation factors or in general we need a measure that tends to generalize over all the different attributes without giving much importance to some of them, also this take into account the correlation that instead **Euclidean distance** do not.

*Correlated variables have a large impact on standard distance measures since a change in any of the correlated variables is reflected in a change in all the correlated variables.*

$$
Mahalanobis(x, y) = \sqrt{(x - y)^T \Sigma^{-1} (x - y)}
$$


![[Pasted image 20231005151130.png]]