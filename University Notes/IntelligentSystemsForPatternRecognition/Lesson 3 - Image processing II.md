Date: [[2023-03-01]]

Status: #notes

Tags: [[Image Processing]], [[Detectors]], [[Descriptors]], [[A.I. Master Degree @Unipi]]

# Detectors

Today another lesson on visual pattern recognition, the other lesson we had problem with invariances, we used *descriptors* like SIFT (Histogram of gradient orientation).

Today : What are the intresting parts or information pieces, that intrest us

## Visual feature detector

There are some useful property we want to have

```ad-note
Repeatability: Detecting same feature in different image portion and in different images

This for these type of changes:
- Photometric 
- Translation
- Rotation 
- Scaling
```

What are the *intrest point* is the foundamental question. Changing the point of view or skewing the images are trasformation *isotrophic*. 

### Edge detector

Edges are rather intresting and because with gradient is easy to find abrubt changes in our images,  these are noisy detectors see the grass in the image below

![[Pasted image 20230301163230.png]]
#### Edge and Gradients

We're gonna convolve our image with a *fitler* that will set black everything that has high response to the image and white the other

$$
\nabla I = [\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}]
$$

direction of **change of intensity**

Edges are pixel where we get abrupt changes in instensity.
We can return a finite difference methods. We will use the **Prewitt** operators that will perform an approximation of gradient but adding an other level of smoothing, they are matrix with our $x$ at the center: eg.

These operators are made from two vectors, one for the horizontal and one for the vertical direction. Multiplying them. One can obtain the way in which you smooth getting different operators.

### Convolving Gradient Operators
![[Pasted image 20230301163252.png]]
And adding more smoothing...
![[Pasted image 20230301163323.png]]
It doesn't even has to be a $3\times3$ matrix, the larger the filter the more is smooth, but if you smooth too much you don't get really what you want. Try it!
Sobel can be scaled also with a factor of $\frac{1}{N}$

```Python
#prewitt mask
kernel_x = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])
kernerl_y = np.array([[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]])

#convolve filters
img_prewitt_x = cv2.filter2D(img, -1, kernel_x)
img_prewitt_y = cv2.filter2D(img, -1, kernel_y)

#show images
```

### Blobs detection

Blobs are connected regions with little gradient variance (*Dots*), an example could be a point on an image, the property of blob is that at a certain point (on the border) there is high gradient and into it we have not that gradient change.
![[Pasted image 20230301164133.png]]

This **Laplacian of Gaussian** (second derivative of the smoother version of the image), we fix a scale $\sigma$ then we convolve to get a responce, the maximum response is then we find a center of a blob with radious of $\sqrt{2}\sigma$ then we change sigma and so on. Because we want to capture different blobs. But the problem will be having more blobs of different size. You could get many responses on a tiny spatial vicinity. Also this vary for scales, we want to decide what blobs we want to keep. Typically if we first suppress on a spacial scale (if pixels are too close), and when you have a maximima across multiple scales you keep that maxima for all the scales.

Thanks to convolution theorem we apply simply the gradient to the Gaussian and then we convolve. Also when you compute the *LoG* you need to use a scale normalized response as we can see above.

### LoG Blob Detection 

1. Convolve with a LoG filter at different scales
	- $\sigma = k\sigma_{0}$ and vary $k$ 
2. Find maxima of squared LoG response

![[Pasted image 20230301165212.png]]
Of course blobs overlaps and we get intrest points in *intrest points* then we proceed to center the *SIFT descriptor* at the center of our image.

#### Pipeline of image analysis

So we use feature detector to get intresting point and then we run a descriptor over that points, that's the typical pipeline of image analysis

Usually the LoG filter can be approximated as **Difference of Gaussian** (DoG) for efficiency reasons

### Affine Detectors

If you scale the Laplacian based detector is invariant to scale but isn't to *affine trasformations*
like skewing image
![[Pasted image 20230301170126.png]]

## Maximally Stable Extremal Regions (MSER)

The idea is to extract covariant regions of an image such that if you change thing they tend to stay togheter, if we perturbe an image we want that certain points are stable to mutation on the image
- Key point: Thresholding image with different and wide range of intensity thresholds and see what points stay close
The blobs are generated (locally), and we get nice properties like:
![[Pasted image 20230301170551.png]]

The same picture has the problem if we apply strong like to a portion of that image, we will substantially change the connected component behaviour.

Intuition on MSER Algorithm:

```ad-note
**Intuition**

Pick a $\theta$ and start to increase the values you get intresting regions, we are intrested in stable regions, such that the change in $\theta$ won't change so much, so we're intrested in a gradient of this thing.
```

![[Pasted image 20230301171223.png]]


MSER Code in OpenCV:
![[Pasted image 20230301171320.png]]

## Image Segmentation

Partitioning an image into set of homogeneous pixels, hoping to match objects or subparts,
images contains segments, that are area with a certain continuity
![[Pasted image 20230301172323.png]]

This above is the results of clustering in attempt to give every pixel a color related to the near cluster of pixel, so something like *K-means* that is a naive approach typically something like $L*a*b$ hoping to cluster togheter regions. In these cases we use different color spaces, we get discontinuous regions.

Today the baseline to do segmentation is **Normalized Cut**

![[Pasted image 20230301172825.png]]

These algorithms take an image and transform it in a graph, and treat it like a graph, we want a weighted graph that represent a certain *affinity* on how much pixels are correlated, the more affine the pixel are, the affinity function could be how close in term of color they are, the job is to identify high affinity and separate those from other pixel that are higher affine between themself, so we want to find an edge in which values on those are low, if you consider this as a cost problem we want  to find the edges that cost less to cut, a minimal cut is a minimal set of edges that you pay less if you cut the components. So instead of minimization this is a min-max problem becuse you cut the minimal edges and keep togheter maximum cost edges, this problem is NP-Complete, fortunately we have a good approximation. 

Resoning in the *Spectral Domain*, a good way to find an orthonormal basis is to find the eigenvectors corresponding to the smallest eigenvalue, picking the **Laplacian** (can trasform adjacency matrix) of the graph, find the eigenvectors and break in two graphs, then repeat on the two sub-graph until we get a stable point. This is the basic idea.

### Pixel issue

An image has a lot of point and we're doing an eigenvectors decomposition of a large image, an efficient trick is to use **SuperPixels** that are agglomeration of pixels, running *K-means*.
![[Pasted image 20230301173829.png]]
So in this way the Laplacian is smaller and take less to complete.

### Take home lesson
Image processing is very much about *convolutions*, these operation are used to apply filter that compute a gradient on an image, masks can apply scaling, computational efficiency is often needed, so for Fourier domain, superpixel speeds up computation. And a good idea is that *random sampling* is a very lightweight feature detector, if you sample enough eventually you get a good approximation.


```ad-summary


```


---
# References

