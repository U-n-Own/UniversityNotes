[[Neuromorphic Computing]], [[Machine Learning]], [[Followed Conferences]], [[ECML]]

Problem: Moore's Law is come to an end and we've reached a Memory Wall

![[Pasted image 20230919165136.png]]

Conventional memories are storing information as charge on a capacitor or trapping charge inside a system, but this require transitor and capacitor and these are not scaling very well, so we need another approach to store information, and to represent it. For example atomic composition of materials: [[Resistive Switching]], associate information with atomic information of device

![[Pasted image 20230919165421.png]]

Applying some signal create a conductive filament so change composition and not charge, we can make it retract, so we get two states a bit!

![[Pasted image 20230919165532.png]]

We can push this further by having different levels and we got a lot of info in one device, also this is nonvolatile!
And these operate at low condactances and low latences.

## Architectural problem

Von Neumann architecture is not the best one for Machine Learning

So with In-Memory Computing we can get the memory inside the computation device!

![[Pasted image 20230919170210.png]]

Equivalent to Von Neumann but adding something new

![[Pasted image 20230919170251.png]]


Static IMC is what we know, the new thing is Dynamic IMC regime, we're expliting non linear characteristic of device each with it's own properties and we can mimic complex functions! We're programming the device depending on the computing.

## Three cathegories


![[Pasted image 20230919170510.png]]
## Computational Bottleneck!

![[Pasted image 20230919170658.png]]

We need to take huge quantities of data move them and performe data intesive Matrix Vector Multiplication

Now we can do this: Kirkoff law to get the current, remember? Natural accumulation!

$G$ is the conductance matrix and voltage vector $v$, apply voltage and read the current in one computational step, so cut down complexity. We can do this with very low power consumption because we just need a signal.

We have high troughput and low efficiency. 

Is important that we can use non-linear phenomena to mimic complex function

![[Pasted image 20230919170757.png]]

## Backprop with inference on this hw

We can do it withou computing all the delta but backpropagation the signal, using intristic non linear phenomena, training neural net inside the memory, this is intresting for Brain-inspired computing, the next section.

![[Pasted image 20230919171317.png]]

## Brain-inspired computing

Real neural systems are dynamical system, and non-linearity is fundamental here.

Common mammalian brain computation happens very differently, no direct signals, non linear structures These exhibit non linear behaviours, synapsis can be potentiated or depressed (thicker or thinner), neurons perform integration, taking signal from dendrites and perform non linear activation and release a spike, these are way more energy efficient, we want to mimic it.

Non-linearity is key!

![[Pasted image 20230919171627.png]]

Eg. Non volatile RAM in order to mimic depression or potentiation. An example is potentiate quickly and then decay in time.

![[Pasted image 20230919172038.png]]

## Emulating human eye behavior

Eyes are connected to visual cortex and by stimulating them we can store a pattern, this is a learning by experience!

How this work?

Well by stimulation in some characteristic way we can replicate this learning, so take an array of devices, initializated at random state eg 0, then feed to system visual stimuly with a bar on different orientation, then the weights (conductances) will start to adapt depending on the signal provided, and this is not programmed, we only show the stimuly!


![[Pasted image 20230919172205.png]]


## Simple Reservoir System

IDEA: input is spatio temporal encoding of image: absence of signal when black, signal up when white, then we fed and start growing a filament inside and more pulse we feed more the filament grow and then the system get to a stable state (i think). Then we use readout to do classification


![[Pasted image 20230919172620.png]]
### Classification

![[Pasted image 20230919173013.png]]

## Accellaration of Algebraic operation

![[Pasted image 20230919173102.png]]

We're changing the direction of the transfer, this is a linear system solution accellerator.

Threats to this are: 
	Stability, by having this feedback loop, since we got analog sensitivity (analog feedback loop has parasitics).

We can create ad-HOC circuits that performs this computations!

![[Pasted image 20230919173442.png]]


### Example

![[Pasted image 20230919173535.png]]


## Page Rank accelleration!

This relies heavily on computation on eigenvector, so we need fast eigendecomposition, and we get nice results. 

![[Pasted image 20230919173615.png]]
## Dimensionality reduction

Eigendecomposition of our covariance matrix needs to be fast, but also energy efficient!

![[Pasted image 20230919173725.png]]