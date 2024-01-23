From Machine Learning course [[A.I. Master Degree @Unipi]]

First: The *Bias* is high if `hypoteses space is small`, not enough powerful model.

*Variance* quantify how the response of the model, if the `hypotesis space is too big`, every time we change training data the hypotesis changes, and if change too much the model is unstable

*Noise*, since the label can have random error if for a given $x$ there are more than one targets. (Imagine a gaussian noise).

![[Pasted image 20230524164529.png]]

Here we can see how the Variance of a high bias changes, this because the data are distributed with a sin function and some $\epsilon$ error, the Variance is plotted by fitting 50 different models by resampling from the training data, where our training data here are the point on the original function we want to approximate by regression.

## Bias-Variance Analysis

Now given a new point $x$, what is the expected prediction error?

$$
E_P[(y-h(x))^2]
$$

Note that there are different $h$ and $y$ for each different training set.