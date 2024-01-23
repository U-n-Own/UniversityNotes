[[Time Series]], [[Data Analysis]]


![[Pasted image 20231220190153.png]]


A time series is composed of parts:

1. Level: average of value in the serie
2. Trend: Increase or decreasing values in the series
3. Seasonality: Repeated short term cycles in series
4. Noise: Coming from randomness...

The first tree can be decomposed from our timeseries but the noise is a non systematic and thus cannot be modeled

## Similarity

How do we define the similarity in time series? 

We can see how two TS are similar at the level of shape and at the structure level

### Euclidean distance

Very sensitive to distortions like : offset of two ts, amplitude scaling, linear trend and noise. We can remove some of them with transformations: like offset is

![[Pasted image 20231220190643.png]]

#### Offset

![[Pasted image 20231220190838.png]]

#### Linear trend 

![[Pasted image 20231220190913.png]]

## Dynamic Time Warping


