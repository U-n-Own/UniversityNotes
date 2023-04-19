Date: [[2023-03-01]]

Status: #notes

Tags: [[Signal Processing]],[[IntelligentSystemsForPatternRecognition]], [[A.I. Master Degree @Unipi]]

# Limitations of DFT
Since we can analyze signal in spectral domain or space domain, we take a signal
![[Pasted image 20230301175010.png]]

If you do a Fourier Analysis on this that has a low frequency component at the start and high frequency component, we want to do the red boxes stuff. This is what **Wavelets** do!

## Graphical intuition
![[Pasted image 20230301175207.png]]
We're splitting the signal into the two domains.

Wavelets are localized in time and space, and are flate in the rest. If we use the *Wavelet transform*, we do it with translation.
![[Pasted image 20230301175321.png]]

What we are doing is correlating the wavelet with our signal in that specific area where the wavelet is not zero, we are also applying it in time by using *convolution* by shifting and convolving, obtaining a response of the wavelet that is localized.

## Wavelets family

There is a generic wavelet formulation

$$
\sum_{t}x(t)\Psi_{j,k}(t)
$$

Terms $k$ and $j$ regulate scaling and shifting of the wavelet, this is a discrete version of our wavelet, to make it continuous you need to integrate between these $k$ (scales) that is called *voice*
$$
\Psi_{j,k}(t) = 2^k/2\Psi((t - k)/2^j)/2^k
$$

- $k < 1$ Compress the signal
- $k > 1$ Dilates it

These are different wavelets you can choose as mother wavelet function
![[Pasted image 20230301180309.png]]

A nice tool to see wavelets is [this](http:wavelets.pybytes.com/)

Some wavelest are defined in complex and other in real, you're not intrested only in amplitude but also in the phase, that is given by imaginary part.

If one picks an higher scale like $2^k$ will get an higher frequency wavelet

![[Pasted image 20230302142846.png]]

When we slide our wavelet trought time and computing convolution, we do it for multiple scales changing it and what we get is something like this
![[Pasted image 20230302142927.png]]

Similar to power density plot in fourier analysis. The $x$ axis will be the space-time and frequency on $y$, so we're getting sorta an intensity of the point, this is a one-shot picture that represent the complexity of our signal. This is kind of a descriptive analysis, also this is a picture.
![[Pasted image 20230302143037.png]]

Usually we have very articulated timeseries, and we can reduce an entier time series to a certain picture like this, then analyzing this picture we can learn.

## Using WT in PR applications

Human activity recognition : we can pickup data that corresponds to the time series, that can be very complicated and long (30 days long time series from human)
![[Pasted image 20230302143526.png]]

This could be done easily with K-means, how? Well you do the Discrete Wavelet Transform of these, then you match the activity you transformed with a database that contains labeled activities and you make the K-means obtaining 0.91 accuracy.


You can as well feed this as 9 channel image to a Convolutional Neural Nets, in this case you get better accuracy
![[Pasted image 20230302143803.png]]


Continuous Wavelet Transform are very heavy to compute but you get very dense representation of data, that on signals are ok, but for images could be very expensive, so maybe is better work on discrete for that, that will get us sparse representation of images.

Wavelets are a very strong relaxation of fourier analysis, they comes with some issues, some of them are good for invertible, morlets are orthonormal basis so we can do inverse transformations.

### Take home lesson
Is important to knwo how they related and differ from fourier.

```ad-summary


```


---
# References

