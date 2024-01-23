[[Data Mining]], [[Classification]]

# Different types of Classifiers

- **Binary vs Multiclass**: self explainatory
- 
- **Deterministic vs Probabilistic**: A deterministic classifier produced discrete-valued label to each data instance whereas a probabilistic one give us the probability to be in a certain class for all classes summing up to 1, probabilistic classifiers are for example *Naive Bayes classifier*, *Bayesian networks* and *logistic regression*, probabilistic provide more information about confidence to assign an instance.
- 
- **Linear vs Nonlinear**: The mathematics is important if we use simple math like a linear separating hyperplane we discriminate instance linearly, but if our data are not linearly separable there is a proble, for example *XOR*, so we need nonlinearity in some way. We could move from a *Perceptron* to a *MLP* by inputting a nonlinearity in the network, or we can *transform* our data but using a function that send the data in higher dimension and then linearly separate them, but we need to be careful!
- 
- **Global vs Local**: A global one fits a single model to entire dataset, unless highly non linear model the one-size-fits-all may not be effective when different attributes relationship with labels varies over the input space, a local classifier partition the input space into smaller regions and fits distinct model to training instances, for example [[K-neaerest neighbor]] classifier is a local one, but being more flexible in term of fitting comples decision boundaries make the model susceptible to overfitting, especially if local regions contains few examples.
- 
- **Generative vs Discriminative:** Given a data instance $x$ a classifier need to predict the label instance of this data, but let's say we are intrested in describing the mechanism that generated the instance, for example the process of classifying if a text is something happy or sad, it may be useful to understand the typical characteristic of the text that involve sadness or happiness. So we can learn the generating process (most usually a distribution) and generate samples from that distribution that resembles the ones we were predicting, so creating new stuff! Examples can be naive Bayes and [[Bayesian Networks]], istead Discriminative include the usual decision trees, rule based, SVM and ANN.
- 
- **Rule based vs Nearest Neighbor**:
# Supervised Classification

As we saw in [[Machine Learning]] course we can have supervised learning when our data are labeled, so we have a certain amount of attributes that are in a certain class. If the problem has categorical nature is called classification, instead if the for the target we have some numerical values associated we do: [[Regression]].
## Rule-based Classifiers
### Decision Tree 

Is a very simple classifier that acts by asking: Having a new example of data eg A new animal is discovered: what specie is? So we do some questions like: Is cold or warm-blooded? Then we do more specific questions to get more insights, so does it lay eggs? Every time we cut the space of the possible lables until we don't have only 1.

The structure of *Decision Trees* is the following

- Root node: This node has no incoming link
- Internal node: This node has both incoming and outcoming links
- Leaf node: Every leaf node has one single label

First two nodes contains the *Attributes test conditions*.

So how do we split? Usually in a greedy manner the algorithms used is *[[Hunt's Algorithm]]*

#### Hunt's algorithm

Tree is built in a recursive fashion and greedy choose.

##### So what is the **Splitting Criterion**?

At each step we must select an attribute to query that will partition the training set into the subnodes

![[Pasted image 20231022162914.png]]

When we split we can merge some of the criterie to keep the branching factor contained.

How do we measure the goodness of a splitting condition?
Well we can measure how *pure* is the split, so how the split divide the example in more omogeneous subsets this is useful because a pure set won't be split anymore and bigger is the set less is the work.

At this point we can speak also about *inpurity* of a set as: how dissimilar are the class labels for data instances belonging to common node.

These are common metrics, *with $p_i(t)$ as frequency of training instance belonging to class $i$ at node $t$ and $c$ total number of classes.*

![[Pasted image 20231022163859.png]]

![[Pasted image 20231022164320.png]]


But this is for a single split, what for keeping track it globally?

Consider a *test condition* that splits $N$ training into $\{v_1,...,v_k\}$ instances of childrens (nodes, so can be a multi split or binary), and then we have $N(v_j)$ specifying how many example are associated with that node and the measure of *Impurity* of that node. So it's a *weighted sum*.

$$
I(children) = \sum_{j=1}^k \frac{N(v_j)}{N}I(v_j)
$$

##### Finding the best splitting

We want to measure the impurity before the splitting namely: $P$ and the impurity after $M$

Defining the **Gain** or $\Delta$ : Lowest impurity measure after splitting $M$

$$\Delta = P-M = I(parent)-I(children)$$

Obviously larged this gain, the better the split, can be also showed that this is a non-decreasing measure, by maximizing the gain we're minimizing the weighted impurity that is our objective, if we use $Entropy$ as measure the gain becomes *Information Gain* ($\Delta_{info}$)

##### Let's see how *Gini index* behaves.

This has a *maximum* when records are equally distributed among all classes, so giving *least interesting information* and has a *minimum* when all the records belong to a class so giving *most intresting information*

![[Pasted image 20231022170632.png]]

##### Let's see how **[[Entropy]]** behaves

Quite similar to gini index, but this tends to give more partitions that are pure, by having more partitions being small. In fact having high number of child can increase complexity and bring to overfit, so we can costraint to get only binary tree by gathering the classes altogether or modify the split criterion to take into account the number of childs produced each time, so we can use this a penalization measure.

#### And for the **Stopping Criterion**?

Usually we stop when all training instances are associated in a node with the same label. We can also do early termination where not all the training instances are labeled the same this can be useful in some cases

## Instance-based Classifiers

