### Author: Vincenzo Gargano

All code used can be found [here](https://github.com/U-n-Own/char-rnn.pytorch) in my fork of the torch rnn.


## Brief recall on RNNs

Recurrent neural networks are best used to model sequences, some practical sequences we've seen are time-series, so signals in time, but also non-wavey sequences data like words in a book, for example Shakespeare works.

- Speech 
- NLP
- Music
- Proteins

Generally when size of input is variable and size of the output is variable.

We've seen `One-to-one` basic *vanilla*, `one-to-many` vanilla unrolled in time taking an input an generating sequence by selflooping the output into itself ,and `many-to-many` (where a sequence goes in and after that ends we get an output sequence) and `many-to-many` bidirectional (sequence in sequence out for each time step).

### RNNs are dynamical systems

From DL Book

$$
\textbf{s}^{(t)} = f(\textbf{s}^{(t-1)}, \textbf x^{(t)};\theta)
$$

We can define our recurrent net like this, where $s^{(t)}$ is the state of our system unfolded in $T$ time steps, and this system is driven by external signal $x$.

![[Pasted image 20230423173738.png]]

Usually hidden units in these nets are defined with this equation

$$
\textbf h^{(t)}= f(\textbf h^{(t-1)} \textbf x^{(t)};\theta)
$$

Unfolding structure will be similar to this

![[Pasted image 20230423174330.png]]




## Backpropagation in Recurrent nets

Again basically backpropagation is the thing we use for learning from errors, changing our weights depending on the error each weights bring to the previous layer until we reach the input. In this case we're unrolling trough sequences *usually trough time*, (sometimes we can unroll in space sequences: images for example). 

Usually in backpropagation we've ground truth the `target y`, we measure the error of our forward pass and backpropagate the gradients of this error in the net (what the error produced against what we expected).

Training is quite simple and what is optimized is a loss over all the time steps (each time step has it's loss), we sum the losses to get the total loss.

### Visualization of knowledge in long-range LSTM cells.

An LSTMs can in principle use its memory cells to remember long-range information and keep track of various attributes of text it is currently processing. For instance, it is a simple exercise to write down toy cell weights that would allow the cell to keep track of whether it is inside a quoted string. However, to our knowledge, the existence of such cells has never been experimentally demonstrated on real-world data. 

In the image we can see what Andre Karpathy did for visualizing what some cells are doing.

![[Pasted image 20230423184344.png]]


#### [Karpathy on Parameters](https://github.com/karpathy/char-rnn#approximate-number-of-parameters): Approximate number of parameters

```ad-note
The two most important parameters that control the model are `rnn_size` and `num_layers`. I would advise that you always use `num_layers` of either 2/3. The `rnn_size` can be adjusted based on how much data you have. The two important quantities to keep track of here are:

-   The number of parameters in your model. This is printed when you start training.
-   The size of your dataset. 1MB file is approximately 1 million characters.

These two should be about the same order of magnitude. It's a little tricky to tell. Here are some examples:

-   I have a 100MB dataset and I'm using the default parameter settings (which currently print 150K parameters). My data size is significantly larger (100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can comfortably afford to make `rnn_size` larger.
-   I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm carefully monitoring my validation loss. If it's larger than my training loss then I may want to try to increase dropout a bit and see if that heps the validation loss.

Or Lercio Headlines dataset is 475kb to choose the best rnn_size and num_layers 
```

## Temperature and Perplexity

*Temperature* is an hypeparameter used to generate new samples, for higher temperature values the net tend to be more "free", so manage to write a lot of different things, instead for low temperature the generated samples are more likely to repeat because mathematically what we're doing is the following thing:

Since neural network produce probabilities using softmax output layer converting *logits*, $z_i$ (aka unscaled output of previous layer) into probabilities $q_i$.

Given a temperature $T$

$$
q_i = \frac{exp(z_i/T)}{\sum_j exp(z_j/T)}
$$
Higher the value of $T$ produces soft probabilities distribution over classes so basically we are doing this $\frac{logits}{T}$, comparing different logits to get a probability distribution

In the code we have this

![[Pasted image 20230428110825.png]]

So getting output_dist and then picking the most probable character from it.

More abstractly we can summarize the Temperature in general for Language Models as the confidence of our model in generating the next sample, higher value of temperature make the model more sensitive to small values this gives more diversity to our generated samples but usually we pay with more errors, like words that do not exists or loosing context information.
Lowering temperature results in the probability of some samples becoming higher, so the probabilities of sequence piece tends to $1$, resulting in much less diversity in our sampling.

*Perplexity* instead is a measure we encountered in the HLT course [[Lesson 2 - Language Modeling#Perplexity]], and basically tells us how suprised is the model on the new data, we would like to minimize the perplexity.

Mathematically we can define perplexity measure like this

$$
PP(W) = 2^{H(W)}
$$

Where $H(W)$ in information theory is the Entropy of our RV defined as $\frac{1}{N}log_2P(w_1,w_2,...,w_N)$


## Char-RNN

What are we trying to do is to learn a probability distribution over our data on "what character should come next?".

## Experiments

For the learning rate i used all the times the same classical 0.01, and Pytorch uses some optimization algorithm like Adam to get an adaptive learning rate.

Let's visualize results of some experiment i tracked using Tensorboard

This was the "generic graph structure" used in almost all experiments

![[Pasted image 20230505180415.png]]

### Experiment 1: Lercio Headlines

I trained two different models one was an LSTM with 100 hidden units and circa 181k parameters

#### Results: Char RNN over Lercio Headlines

During training temperature was setted pretty high: 0.8

What i was during training was that initially, after first epochs, the output is pretty poor, the model doesn't undestand where to place letter and in what order, after a while the model is capable to put the quotes at the right spot, so at the start and at the end of a phrase or a sentence. Or maybe sometime when citing someone speaking. Using the constructs `Name: "citation"` or `Subject:"news about it"`

![[Pasted image 20230423233942.png]]

---

#### LSTM: 

A more refined model output changing the temperature (lowering it).
![[Pasted image 20230505184556.png]]

Again let's try with high temperature at the end of training : temperature set of 0.7

![[Pasted image 20230405021019.png]]

As we can see the model is making up new words, this is because of the char-by-char generation, temperature make it more creative but sometimes we want that tradeoff between creativity and fidelity to data. From what i sperimented the ideal temperature reside in something from 0.4 to 0.5.

---

#### GRU:

Low-medium temperature:

![[Pasted image 20230508195329.png]]

Higher temperature:

![[Pasted image 20230405021409.png]]

Reading these generation i expeted better results from

### Experiment 2: Drosophila Melanogaster Genome

Model used here was an LSTM with 64 hidden units over 250 epochs, for a total of 79460 parameters.

![[Pasted image 20230505175938.png]]


#### Results : Char RNN generating drosophila genome and aliging with BLAST

Result I got by sampling from the CharRNN, with low temperature: What i done is generating some sequences then feed them to the Protein\Genome similarity search website and wait for results. The metric used by this website to quantify how much similar they are is the E-value, which evaluation depend from the dataset and the lenght of the data user gives. Smaller the value better the similarity, what i think is happened here is that i picked a small piece (500 character sequence) and fed it to the database, what i got is a very similar piece to the `chromosome 3L` of the Drosophila, i tried with other high temperature and i got a similarity with some part of the entire chromosome X, but the E-value in that case was pretty high: 0.041.


![[Pasted image 20230509125308.png]]

### Experiment 3: Music generation


Here i searched for the most simple and big dataset i could get my hands on, i tought to convert a large portion of midi to text, but didn't knowing really how to do this in the current time i just took [this dataset](https://github.com/cedricdeboom/character-level-rnn-datasets)

This time i tried both a LSTM and a GRU, and i obtained nice results with the GRU!

#### Results:

I just asked the net to generate the sequence of notes then i wrote a little script using a library called [SCAMP](http://scamp.marcevanstein.com/) to generate the scores.
I just added dynamic like loudness and duration.

This sample was generated with 0.5 temperature and the start is quite messy, but then further it became pretty nice the melody towards the end.

![[Pasted image 20230512025059.png]]

Unfortunately on pdf i can't insert sounds here.

Now let's try with a melody i use as prime string to the net and just lowering the temperature

![[Pasted image 20230512031119.png]]

And what if we crank up the temperature? 


![[Pasted image 20230512031605.png]]

Yeah the piece becomes more jazz than classical music. But there are certain parts where the GRU come up with some octaves that's really nice, its in the 3 measure.

## Interpreting results with Tensorboard

Some plots of our loss, gradient and weights histograms from Lercio training.

![[Pasted image 20230429215617.png]]
![[Pasted image 20230429220007.png]]

---

Let's see the gradients and weights at the **start**  and **after a little** while of training of the train on another model and different data.

![[Pasted image 20230405023836.png]]

After some epochs...

![[Pasted image 20230405024054.png]]

---

![[Pasted image 20230405023803.png]]

After some epochs...

![[Pasted image 20230405024013.png]]

So by exploring these plots i understood that initially the distribution are more irregular but still near a normal distribution, probably because of initialization, then these tends to become rounded or generally tuned toward the training process, since in the RNN weights are shared trought time steps. 

For the gradients we we oberve a spike, these can we observe that are greater at the start and becomes smaller **"more spiked"** so narrowed their values because of the optimization making progress for the learning.

### Conclusion and future works

RNNs, but more intrestingly LSTMs and GRUs are capable to work on sequences, and we have a lot *different* data sequences that can feed them, characters in language, musical notes, DNA and Protein sequences and many more. What we can expect is to draw new but, a lot of time nonsensical samples, from a distribution that is mimiking our original from what the data we feed them.

What if we want to make generation more affine to our language? Well we need to give some **context** information to our network, so that it understands the connections between the words, and this is what the **Attention** mechanism do!
Basically what attention do is to improve the encoder-decoder by allowing decoder to utilize the relevants part of the input sequence by *weighting* the encoding input informations.

And finally we can talk about the limitation of these RNNs, widely known is the slow and unparallelizable training of these sequential model that gave rise to the Transformers architecture. More is that very long term dependancies are lost and we can encounter problems like **vanishing gradients**.
 
But still these models are powerful for modelling sequences especially if time is involved in fact transformers uses *positional encoding* to deal with the temporal order in input sequences. Also RNNs have some internal memory and can work with information more "naturally" than the new architectures, they are more simple and more similar to how biological brain works with retroaction of the signal that comes back as input.


### Resources

[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) blogpost from which Andrew Karpathy explain and reason about how recurrent networks... 

[From Visualizing And Understanding Recurrent Networks by Karpathy](http://vision.stanford.edu/pdf/KarpathyICLR2016.pdf)
[Temperature explained by Hinton](https://arxiv.org/abs/1503.02531)

Drosophila experiment website for segment similarity

```ad-faq
[Protein Similarity](https://www.ebi.ac.uk/Tools/sss/fasta/)
[Genome similarity](https://blast.ncbi.nlm.nih.gov/Blast.cgi)
```

