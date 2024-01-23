[[Statistics]],[[Probability]]

## Z-score vs T-scores

These are two different statistics, namely the Z-statistic will tell us how many standard deviations $\sigma$ away we are from the sample mean.

$$
Z \text{-} score = \frac{\overline{X} - \mu_x}{\sigma_\overline{x}}   
$$

We can assume by **[[Central Limit Theorem]]**: $\sigma_\overline{x}$ = $\frac{\sigma}{\sqrt{n}}$. Also we don't know the true $\sigma$, we will use the sample standard deviation : $\color{yellow}{s}$

This is true for $n \geq 30$. 
$$
Z \text{-} score = \frac{\overline{X} - \mu_x}{\frac{\color{yellow}{s}}{\sqrt{n}}}   
$$


Example : What is the probability to get that extreme result?

![[Pasted image 20231027183713.png]]

Now if the $n \leq 30$ it won't be normally distributed then this will have a **T-distribution**. With a $\mu = 0$.

![[Pasted image 20231027184749.png]]
