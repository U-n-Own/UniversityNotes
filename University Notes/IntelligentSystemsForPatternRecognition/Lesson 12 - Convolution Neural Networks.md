Date: [[2023-03-23]]

Status: #notes

Tags: #ispr[[A.I. Master Degree @Unipi]]

# More generative Graphical models

Generative models are good at unsupervised task, but weak on predictive performances, we've seen Bayesian Nets, then we've seen MRF if we've some prior knowledge to use for example features funtion. These are computationally heavy usually.
Finally dynamical models that model relationships between the data unfolding in time or trough the data structure, modeling also complex casual relationshop.

![[Pasted image 20230323142446.png]]

Consider avoiding generative models for time constraint or computational constraint or noisy data, these have very high variance we don't want.

We've seen variational inference and sampling, the first don't approximate trough the real distribution instead the second is more precise more samples we draw.

*Neural nets* are better in interpretability, so we can use them as *variational function*, they can be also used to sample.

# Deep Learning
![[Pasted image 20230323142753.png]]

# CNN - Part 1

Let's decompose how CNN works, dissecting in it's components. We are going to see at convolution under some details, what they capture.

### Some story

CNN aren't born in 2012, also neither before in 1944. The idea of this architecture comes from this ***Neocognitron***, from a Japanese professor. This guy read paper from those who won nobel prize for study of visual cortex on model of neuron, he tried to put it in how the brain process it: these cells simple and complex detect different features.
- Simples : edges, blobs etc...
- Complex: receive input from simples and combine them and respond to certain edges, this responds to all the edges, this is a convolution + pooling operation. This was trained by unsupervsed learning, at the time there wasn't backprop and these are really a lot of layers. Going from simple to complex was a hierarchical work and was even repeated by extracting incrementally abstract representation of our informations.
![[Pasted image 20230323143839.png]]

### First CNN for sequences
![[Pasted image 20230323144326.png]]

These 15 time steps are taken by 1-dimensional convolution, after that we have a shrink and another convolution until we get a very simplified version and spit out the output, this was trained by backprop + parameter sharing.

This is called `Time-delayed Neural Network`

### CNN for images
![[Pasted image 20230323144629.png]]

This was one of the first CNNs by Yan-LeCun, this was inspired by complex and simple cells mechanism. This couldn't scale!

### Bio inspired model HMAX
![[Pasted image 20230323144823.png]]
This was a model builded on a complex-simple cells constructing detector of features, these was bioinspired by a certain points with a central idea using SVMs. There were a lot of layer so backpropagating was really difficult.


### Dense Vector Multiplication

If we consider images as vector we're destroying images, this is an example with a simple joke image
![[Pasted image 20230323145207.png]]

The bits on the image are disconnected pieces of image, we're just destroying information. If there is a neuron that highly responds on that bottle, if we shift the image this isn't even captured, so that neuron won't recognize the bottle, this isn't translation invariant. 
There are case in which we want dense image.
What is an example? If we have a sensor that hitten by particles, using a dense approach is more adapt, if we want to capture a specific geometry convolutional approach are suboptimal.

### Convolution operation

![[Pasted image 20230323145614.png]]
We've seen it at start of the course, we're going with our filter convolving on the image and extracting a values responding to certain part to respond.

How convolution is used in NNet?

If this is a image we can have a *neuron* as a masked part of this image the matrix $3\times3$. That matrix is our weight.
![[Pasted image 20230323145744.png]]
We're going to get a smaller images after applying convolution

For color we add 3 channels for capture colors. So we have tensors $32\times32\times3$.
![[Pasted image 20230323150129.png]]

So we get a convolution of the channels.
![[Pasted image 20230323150231.png]]

### Stride
This is the measure of jump of our filter through the image, the bigger is less multiplication we do but we're doing basically subsampling, if we have high resolution images we can add an higher stride because doesn't matter. There is a general rule to calculate the new size.

Given $W$ the lenght of the columns and $H$ for the rows in the image, $K$ will be the dimension of our Kernel and $S$ as the stride.

![[Pasted image 20230323150510.png]]

But we don't want to lose information when convolving, so we can do something called padding, surrounding the images with a borders of zeros
![[Pasted image 20230323150604.png]]

What are we really doing? A linear operation and we are working with neural nets, so we need a non-linearity, the effect of adding non linearity added pointwise is that, typical thing used is ReLU. Or it's variations.
![[Pasted image 20230323150701.png]]

### Pooling

The first part of a CNN is *Convolution*, now we talk about the other type of neurons. The Pooling one, after we apply a convolution we apply a "pooling". To make robust our model to some transformation the idea of pooling is to get to know the same features with slighly different manifestations. The subsampling that is done by pooling give a more abstract representation, at low levels we are responsive to low level feature (edges) at higher level we want to make model respondent to object (made of feature like different edges). Making our image more smaller is a certain feature, this happen only if the feature fell into the entire convolutional filter. So a way to speeding up the operation to recognize, another way is striding but that make you use parameters while pooling has not parameters.
![[Pasted image 20230323152328.png]]

Usually the pooling we use is MaxPooling, taking the max between the four subparts. The maxpooling operate over a slice, if we have more slice we can operate maxpooling over channels. 
Alternatives are
- Average pooling
- L2-norm pooling
- Random pooling

Dimension after pooling
![[Pasted image 20230323152552.png]]

The term convolutional layer is referring to this
![[Pasted image 20230323152618.png]]

### How do we work with bigger picture level
![[Pasted image 20230323152721.png]]

Convolutional layer are what we discussed, when image shrinks there is a pooling operator before usually. 
Why the layers thinkens? Well if we have more than 1 filter we will have each filter with some parameters, with thickness of CL 1, thickness $D$. It thickens more the we apply conv layers, $D'$ in CL2 isn't by chance

$$
D < D' <D'' < D'''
$$

Why this happens? At low level we dont have a lot of features, at the start we have conv filter that capture blobs, edges that point in certain directions, we are making informtion smaller, but more complex because we are not loosing information. We are extracting features. At the end at the convolution part we're going from a full feature collection to a single neuron. And finally one single output, there is an issue in the part from the last CL4 to the fully connected layer, here we have one parameter for each input this is a shitload of stuff, and this is just a neuron. This is the **bottleneck** for CNN here weight sharing is lost, this motivated other architectures that.

### Convolutional filters banks
The resulting dimension on these images using filter $K\times K \times D_1$
![[Pasted image 20230323153738.png]]

### Final note on convolution
![[Pasted image 20230323154108.png]]

This usually what convolution is implemented, this is more like cross-correlation and not convolution.

# CNN part 2

## CNNs as sparse NN

Let's say we have a certain number of conv neurons, if we're in the part $x_2,x_3,x_3$, first thing that is a sparse perceptron neuron getting input from 3 different neurons, this has another intresting effect: *weight sharing*!

Sparsifing connectivity in our network and sharing weights is the key of CNNs!

So for $s_3$ we have $x_{2}c+x_3b+x_4a$.
![[Pasted image 20230323154231.png]]

This make us understand that our weight is responsible for more, and we're gonna sum the gradient for each of contribution when we're doing backprop. That's how backprop works there. 

### Strided convolution

We start jumping in this part, making is sparser.
![[Pasted image 20230323154600.png]]

### Pooling or MaxPooling and spatial invariance


We can se that the receptive field the second in pooling, what we're gaining in pooling layer is the responses of the feature map in the receptive fild (collection of arrows that point into the pooling layer).
![[Pasted image 20230323154640.png]]
Pooling is behind the idea of invariancem with this layer of invariance we will be more robust with respect to invariance, then if we do cross channel pooling we can be really invariant, as we can see the $5$ oriented in different way. Pooling unit responds to 5 independently of the rotation of the image.
![[Pasted image 20230323154854.png]]

### Why layering is good?

We can see by this image, the deeper we go into the layer, the more neurons up will have a broader vision.
![[Pasted image 20230323155024.png]]

## Training CNN

![[Pasted image 20230328111921.png]]

What we would do is to compute gradient with respect to a $\Delta w_i$ and summing contributions from all the connections. You need to backpropagate gradients, we don't have a full weight matrix, but we can recreate one by replicating one parameter many times, because of the convolution filter leave us with less stuff.

![[Pasted image 20230328112126.png]]

We can see more than 1 $w_{0,0}$ but creating this matrix is really a waste because we have a lot of zero multiplication how do we do?

## Deconvolution (Transposed convolution)

A way to go back from convolution, idea is simple: call $A$ the blue thing and $B$ the green, $A$ is the original image, what we see at right in green is the convolution. So we want to go from $2\times2$ image, how do we do? Use anothe kernel $3\times3$ and adding some padding.
So deconvolution is just another convolution. The kernel use for deconvolve is different from that we use to convolve, so we have convolution and deconvolution layer, with different parameters.

This is an example with zero padding
![[Pasted image 20230328112305.png]]


What if we have some padding and we add another stride, you're not convolving every pixel but only jumping. So at right we can see row and columns of padding.
![[Pasted image 20230328112740.png]]

## Architectures

LeNet-5 was a first architecture from 1989, trained on MNIST data
![[Pasted image 20230328112950.png]]

Subsampling in this case is maxpooling, pooling wasn't performed cross-channel but only on the single feature maps, then every neuron in the fully connected layer is connected to each of the matrix (16 matrix 5x5) entries. So a lot of parameters. Then we apply a nonlinearity a sigmoid. This couldn't be scaled obviously.

## Image net competitions

Before this people were doing thing using SIFT descriptors, CRF and other stuff, then this broke competition. Size and complexity is a lot bigger than the previous, also this is split in two, half of images processed on a GPU and half by another. Final images were 13x13 for 128 feature maps and connected to 2048 wopping FC neurons. We have data augumentations like rotation, random crops etc.

They introduced the use of ReLU good for trainability
Also there were a dense regularization by dropout.
![[Pasted image 20230328113602.png]]

### ReLU non linearity

At the time was used sigmoid but sigmoid had a thing towards the start and end with saturation because gradient there becomes zero and this brings to vanishing gradient. ReLU is really simple, first derivative 1 and 0 elsewhere, second derivative zero, no second order effect. Sigmoid are nice but only for final layer, using ReLU grants that the gradient of a linear function to be positive everytime. The part of zero gradient there will create dead units, for regularization mechanism but we would like to control that. SeLU for example.

```ad-note
For example we don't like ReLU for RNNs because they are likely gonna explode in gradient, we usually use tanh for that because of the nature of the function that tends to not grow linearly but stop at a certain point.
```

![[Pasted image 20230328114220.png]]

Alexnet
![[Pasted image 20230328115126.png]]

## VGGNet
![[Pasted image 20230328115158.png]]
This didn't won competition but reduced by half from the last we saw, this was an attempt to standardize CNNs models, this was VGG-16   where 16 are the layer of convolution. This was by Oxford

## GoogleLeNet (2015)

There was scalability problem, and one type of convolution they get rid of dense layer and differentiate size of filters
![[Pasted image 20230328115444.png]]


The 1x1 convolution are shrinking the channel by one flattened image, that will be done on the 3x3 and 5x5 on a single image because the previous layers could bring a very high number of channels. Also the yellow part are used to reinforce gradeint layer because for these deep structure the gradients dies at a certain point


### Batch normalization

What's this technique?
The original intuition is that if you're training with minibatches is to pick certain number of samples compute all these and then update gradient, change parameter then take another minibatch, what if between two batches the parameters change? The distribution of activation of neurons depends on the neurons on the layer afterwards. Distribution activation in layer $l$ are a certain distribution $D_l$ a distribution in the layer $l+1$ instead can have a different distribution $D_{l+1}$ this can create issues, can converge later. Would take more and more steps
![[Pasted image 20230328120327.png]]
What we do is substract the mean and divide by variance of the minibatch just computed, but we cannot impose normalization without allowing the network to cancel what we've done the trick is to do the scale and shift, $\gamma$ cancels out the effect of the standard deviation and $\beta$ remove...
$\gamma$ and $\beta$ are ours parameters that will change trough backpropagation. These are used to counteract *covariate shift*.

## ResNet

With ResNet we get 152 whopping layers, how gradients survives? The skipping connection or jumping connection, shortcuts (*Residual connections*).

After the jump we pass over the original input at the start, so that we focus on what was different on the input. 
![[Pasted image 20230328122250.png]]

What we do basically on this is just use the input and sum it with the convolution, composing $F(X) + X$, when backpropagating the gradient can flow (for example for the identity function we have backprop 1 in the residuals). What that block do is to compute the residual from the convoluted and the passed input stuff. This is connected with Dynamical Systems these are a certain type of dynamica system, these connectivity are non-dissipative and don't degenerate information. This helps a lot with gradient flow to help make it don't die.
![[Pasted image 20230328122556.png]]

## MobileNets

Low powered device like mobile phones, and since there are too many multiplication.

Here we have $D_{K}\times D_K$ filters for $N$ where is the size of original image, what we can do is separate depthwise multiplication from spatial multiplication by using the same trick google used by multiplying with 1x1 filters, to simplify stuff.
![[Pasted image 20230328123020.png]]


## CNN Architectures evolution

![[Pasted image 20230328123358.png]]

### What can we do with CNN

What representation CNN are learning, we're projecting our high dimensional fitler generation in two dimension in another image to understand what the neural network learnt. So from final layer to two dimension eg:
![[Pasted image 20230328123723.png]]

What for intermediate layers? (Hidden)

Visualize kernel weights, filters using naive approach that work only for first layers that show us blobs, lines and so on.
Or we can *map the activation of the convolutional kernel back in pixel space*, so deconvolving. We can also see how a filters responds.

### Deconvolve nets
![[Pasted image 20230328123955.png]]
We can deconvolve a specific layer, just attach from that layer to the specular layer and apply the chain of deconvolution to the original place and see what we get!

![[Pasted image 20230328124321.png]]

### Filter responding

Or we can as we said earlier how filter responds to some part of images, for LAYER 1
![[Pasted image 20230328124350.png]]

Layer 2

![[Pasted image 20230328124505.png]]

Layer 3

![[Pasted image 20230328124526.png]]


### Occlusions

We can see how higly respondant are on certain part of images the neurons.
![[Pasted image 20230328124814.png]]


### Causal convolutions & Dilated convolutions
There is now future in the first hidden layer it depends on the two under it, after 3 layer we can view only 4 input, can we do better?
![[Pasted image 20230328124855.png]]


Well we can convolve and jump!

*Dilated convolutions*: Similar to stride but we preserve the size of the image, by using simpler convolutions
![[Pasted image 20230328125021.png]]

## Semantic segmentation

Given an image i want another image that segments it, we can't do this for convolution, a simple way of doing this is to use fully convolutional NN
![[Pasted image 20230328125150.png]]

But this is really bad, too many parameters!
![[Pasted image 20230328125314.png]]

## Deconvolution architecture

How we backprop trough pooling? That's not differentiable
![[Pasted image 20230328125359.png]]


### Stacking and visualizing dilated convolutions

Substantially better results no fragmentation of areas because receptive field always know if a pixel in middle of images is in another class.
![[Pasted image 20230328125845.png]]

### Convolution 1D for genomic sequences
![[Pasted image 20230328125956.png]]


### Take home lesson

```ad-summary


```


---
# References

