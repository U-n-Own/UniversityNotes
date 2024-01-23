[[2023-02-27]]

Status: #notes 

Tags: [[Ispr]], [[Image processing]], [[Signal Processing 1]], [[A.I. Master Degree @Unipi]]


# Images are tensors!

Today we will talk about images, how can we find spatial and `spectral` information in our images? Yeah we can do frequency analysis on those.

Let's start by talking how images are treated in computer word, so we have different cases a *grayscale* image will be a matrix with our pixel (if normalized) whose value goes from 0 to 1 and become more white towards 1. But what for color images? Well usualy $(R,G,B)$ images have three channels per pixel where we can express what colour is that specific pixel so this is the reason why the images are matrixes with channels, basically we call these tensors.

But the colors space of $R,G,B$ isn't the only one we can use, in other cases other color spaces are more useful, for example $CIELUV$ color space add something that indicate lightness, with combination of blue-yellow axis and gree-red axis. We can move linearly in this perceptual color space.

What task are we intrested with images?

- *Identification*: Region of intrest (low level features) like cornes, lines, blocks, blobs...
- *Classification*: More abstract and high level task we want to find certain objects, we can have different type of classification task life obects in an images, or multiple objects. To do this we need to rely on indentification of the low level feature by deomposing complex items.
- *Segmentation*: Coherence in the image based on a certain metric. Dividing it in parts based on those characteristics.
- *Semantic segmentation*: More like the previous one but we want to add some labeling to pixels and regions.
- *Automated image capturing*: Use classification to generate sentences that describe images.

## How to represent visual information?

When we take an image we have raw signal that could be not useful, some other properties instead we want to maintain.
We want to be informative don’t throw away useful information and maybe only throw redundant, also Invariant, for example if you move but take two picture (these have same informations), or take pictures with lights on or off; these variations are everywhere so we don’t want to have descriptors to be weak towards these changes, some transformation like rotation that make the object different. We want also it to be efficient, bigger the image more the pixels so... querying and indexing is difficult.

## How to spot informative parts?

Represent distribution on the whole image, like colors, edges, corners: What are the main colors(discriminating bewteen the distirbution of color), distribution of edges. Histograms are very good to represent distributions, a classical one could be number of pixels of a given color, but having histogram of a color what really
tells us? You can transform the image. What if i’m instrested to get info on certain part of the image, for example background made of grass, my descriptor will be less noisy if it’s local. *Approaches local is more informative* but things you need to reason is how do i find intresting points and how do i’m sure that extracting info on these is compliant with differences, different location, rotation and scaling of the item/thing that intrest us in the image.

Most of these are approached by **convolution** with some filter. A localized *descriptor* can be a patch of my image, i linearize it in a vector, so i have destoryed the information, this isn’t invariant wrt intensity so no good descriptor. No invariance to scaling, rotation, pose...

![[Pasted image 20230610175054.png]]


If i rotate image, i get different representation from my patch! These pixel based representation don’t works. I want to represent my pixels exactly and that if i scramble the pixels i don't lose . If i randomly permute all pixels in an image the histogram remains the same i had before scrambling! 

Represent patch by histograms describing properties, a simple approach to have histogram of pixel intensities on a subwindow but this isn’t invariant enough, but we want invariant to illlumination (normalize it), scale (captured at multiple scale by simulating) and more


# SIFT

**Scale Invariant Feature Transform**, it’s a descriptor that is invariant to scale.

`Idea`:
1. Center the image patch on a pixel $x, y$ of image $I$
2. Represent the image at scale σ, control how close we look at an image. If we change $\sigma$ we blur or sharp the image, like a magnifing glass or mathematically speaking a gaussian. So we convolve the image with a Gaussian filter $G$, the larger the gaussian the blur is the image, the smaller is gaussian less are the other pixel that contributes!

### Gaussian filter

Basically doing local averaging, larger the variance, more pixel will contribute to rewrite the value, so larger variance, blurred image, less pixel sharper image. Smaller differences will be pushed up in a certain way. 

![[Pasted image 20230610175901.png]]

We’re in a discrete word so defined over discrete set, we create a grid then we run a gaussian filter on these grid that will become our patch, running it on our image will get us a convolution that with σ larger than 1 will blur instead with a small than 1 will sharpen the image. Basically on the ridges you can see very well when you apply the gaussian with low σ the black area will become very black and white will become more white and the border will be more visible. 

3. Next we can see how much we change from dark to brighter, we want the *gradient of intensity in the patch*: (`Direction of changes in luminance`), consider a magnitude $m$ and orientation $\theta$, using finite differences, so you’re in an image and fix $x$ and look at $y + 1, y − 1$ and the same with fixed y, convolution we can simplify with this simple filter. So the dark area becomes darker and white are whiter, so more responsive in areas like ridges, so the change between dark-white area are more evident. So what we do with SIFT is to convolve image with Gaussians at different scales, and we will represent it in an *histogram like*.

![[Pasted image 20230610180145.png]]

We can simply this by using these filters and convolving.

![[Pasted image 20230610182057.png]]

This is what we get 

![[Pasted image 20230610182154.png]]

So we want the direction of change, if we see where the intensity grows

4. Calculating the gradient for each sigma, we create a *histogram gradient*

This is our imagem what we try to understand is the direction of the intensity where grows, we subdivide images into regions, and we compute gradient for each of the 4x4 regions, so from 16 pixels we get the average orientation becoming 8-long bin histogram. We repeat this for each of the 4x4 region are we get 16 8-long bin histograms: 16x8 vector: 128.

## SIFT becoming rotation invariant and more

SIFT isn't rotation invariant, what can we do?
Computing average gradient in the subpatch. So we have a leading gradient the one at center and we calculate the other based on the one at the center, so if image rotatates leading gradient rotate but differences related to that are the same!

![[Pasted image 20230610193004.png]]

## Love the Gaussian

Scaling our gradients based on the gaussian *red circle*
over the image, so that some elements far from centered pixel is more important than thing less important. In this 16x8 descriptor the items that matter the most are gaussianly nearer from our focussed pixel!

To achieve invariant to luminance this can be useful.
So we threshold.

## Generally SIFT is:

A 128-dimensional vector of gradient directions that encode orientations of the image patches using subpart of these patches, by running it on an image we get intresting points where these gradients change abrubptly

# Fourier Analysis

Let's look at the spectral dimension of images. 

Images are function returning intensity values $I(x,y)$ on the 2D spanned by $(x,y)$. The function is the intensity value.

$$
\mathcal{F}(f ∗ g) = \mathcal{F}(f ) ∗ \mathcal{F}(g)
$$

Transform convolution in elements-wise multiplication in Fourier domain, suppose an image $I * g = F^{-1}(\mathcal{F}(I) ∗ \mathcal{F}(g))$, where $F^{-1}$ is the inverse fourier transform, so we can see the convolution is the inverse of the multiplication in Fourier domain 

Why useful this? Some stuff is easy to do in spectral domain that are not easily computed in spatial domain, and we can then invert fourier.

For example we can take a graph and define a CNN on a graph, and instead of convolving on graphs, (on image is easy because pixels are ordered), but ordering in graph is NP hard, but if we fourier decompose graph and convolutional filter, we can do it by operating in spectral domain.

Summing into two direction we get this

![[Pasted image 20230611173558.png]]

Composing sin and cos to create an image!

![[Pasted image 20230611173621.png]]