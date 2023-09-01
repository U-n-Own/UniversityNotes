Date: [[2023-04-26]]

Status: #notes

Tags: #hlt [[A.I. Master Degree @Unipi]]

# Structural Ambiguity

![[Pasted image 20230426142917.png]]

![[Pasted image 20230426143127.png]]

What are we intrested is *Syntax* and *Grammar*, practical uses of parsing is to extract knowledge relation, semantics (protein interactions), Sentiment analysis, Summarization (what parts are more important and what we can discard?) and Question Answering.
Also Translation and Negation (change direction of opinions).

### Sequences miss long term dependancies - summatization

![[Pasted image 20230426143313.png]]

The second sentence show that isn't only important to group sentences because summarization info are scattered in the text

### Dependancy paths identify relationships

Protein interactions: How KaiA,KaiB and KaiC interacts with SasA

![[Pasted image 20230426143444.png]]

### Question answering

Parsing sentences for question answering

![[Pasted image 20230426143543.png]]

![[Pasted image 20230426143615.png]]


### Chomsky Grammars

We've seen these context free grammars that are those that constitute computer programs for example, where we have tree to describe how a certain part of the program behave by using production rules of our grammar

![[Pasted image 20230426143800.png]]

### CFG

| Symbol   | Meaning                                      |
| -------- | -------------------------------------------- |
| V        | Set of non-terminal symbols                  |
| $\Sigma$ | Set of terminal symbols                       |
| R        | Set of rules or productions                   |
| S        | Starting symbol or axiom                      |

A context-free grammar can be defined by four components:

1. **V (non-terminal symbols):** This set contains symbols that represent variables or placeholders for strings of other symbols. These variables can be replaced with other strings using production rules.

2. **$\Sigma$ (terminal symbols):** This set contains symbols that cannot be replaced any further, they are the base elements of the language.

3. **R (rules or productions):** This set contains rules in the form of V $\rightarrow$ ($\Sigma$, V)$^{*}$. That is, each rule maps a non-terminal symbol to a string of terminal and non-terminal symbols.

4. **S (starting symbol or axiom):** This is a single non-terminal symbol that represents the initial state of the grammar, from which all possible sentences in the language can be derived.

Together, these components define a context-free grammar $G(V, \Sigma, R, S)$ that generates a language consisting of all possible strings that can be derived by applying the production rules starting from the starting symbol 

### Statistical parsing

![[Pasted image 20230426144425.png]]

## Dependancy grammars

Dependancy grammars turned out to be good for eastern languages like Czech, Russia, Chinese but these languages words have free orders, or in other ways the order isn't strict, other grammars impose strict context. So the idea is : instead to use tree structure we use relation between words directly. In dependency grammar, each word is connected to its governing word or *head*. This creates a network of relationships between words in a sentence, rather than a hierarchical tree structure. This allows for greater flexibility in word order and more accurately reflects the way these languages are spoken.

Dependency grammar also has the advantage of being able to handle complex sentence structures and providing a clear representation of the relationships between words in a sentence. 


![[Pasted image 20230426144747.png]]

### Annotation criteria

![[Pasted image 20230426145051.png]]

#### Tricky cases

![[Pasted image 20230426145228.png]]

### Dependancy tree

More than a simple tree is a *directed rooted tree* weakly connected

![[Pasted image 20230426145507.png]]

For example, `He` the subject of `design` and `develops`, connecting to them but develops is not directly connected to He.
![[Pasted image 20230426145602.png]]

Another difficults case is this in italian `che` = `who`

So `che` is the subject of `cercheranno lavoro` but also the part before `coloro`. We can't put two links on the same node, what is the solution?

![[Pasted image 20230426145721.png]]

Solution to these are: for second example we use indirect links like `che` reached by `cercheranno` and `coloro` reaching `cercheranno`.

![[Pasted image 20230426145845.png]]

### Annotated corpora are used for computational liguistics

![[Pasted image 20230426150024.png]]

## Dependancy parsing

There are several technique to do depedancy parsing

In *graph based* consider all the possible dependancy graph and choose the one with the higher probability.

In *transition based* parsing, the parser builds a parse tree incrementally by adding one word at a time, based on a set of predefined rules or actions. Theparser starts with an empty stack and a buffer containing the input sentence. At each step, the parser chooses an action based on the current state of the 

![[Pasted image 20230426150133.png]]

```ad-cite
Best way to learn something is invent it. 
```

Le'ts invent a dependancy parsers, so parser should decide what word are connected with each other, dependancy parser take look two words at a time: *top* and *next* words. Imagine to have them in a stack, and the remaining words are in a queue, taking them from the queue

Are top and next related? Yes, they are related, who is the head and what is dependant? 
The subject could be the head or the verb. Usually the verb is the most important, so we create a link right to left : head to dependant. So `He` becomes the child of `saw`. Then we shift such that `saw` will be the "new" top and next will become `a`, between `saw` and `a` there's no relatioship directed so we do not create an arc, and we shift again. Now top word becomes `a` and the next becomes `girl`, by doing this `girl` is the most important and `a` becomes child of `girl`, then the pointer switch to top on the stack of `saw-he` and next points to `girl-a`, so what's the most important? Still `saw` we create a right arc such that out tree like structure becomes `saw-he,girl-a` after this we shift so top is `saw` and next is `with`,  but `with` to who refers? To `he` or `girl`? We must be clever. We could go back to the state where top and next were `saw` and `girl` and do something else, the parser need to choose if he have to shift,left-arc,right-arc. We can try to use [this simulator](http://medialab.di.unipi.it/Project/QA/Parser/sim.html)

![[Pasted image 20230426153806.png]]

![[Pasted image 20230426152037.png]]



### Shift/Reduce Dependency Parser

Training a Parser: how do we train a parser, he needs to learn to do three action, systems that have to learn to apply 3 rules are basically classifiers, so we train the system to do classification taks and choose the most likely one.

### Dependancy graphs


![[Pasted image 20230426153950.png]]

### Parser state 

Parser moves trough states, we can see the inferenc rules above

![[Pasted image 20230426154215.png]]

![[Pasted image 20230426154117.png]]

### Parser algorithm

A fully *Deterministic* algorithm using trained model to predict next rule to be applied, giving a representation of current state

![[Pasted image 20230426154340.png]]

Where the selectAction uses a trained classifier and getContext takes the context (representation of the state), the simply we take the more likely action that is choosen by classifier.

#### Oracle

Performs steps and knows what is the exepcted results, the correct answers and try to achieve it, modifying the behaviour of the parser.

##### Arc-standard oracle
Another condition: the node used as a chield should not have others childs: eg connecting `girl` with `telescope`.
![[Pasted image 20230426154731.png]]

### Projectivity

Is an issue, a property of the graph, it can be written without any crossing between words

![[Pasted image 20230426155026.png]]

eg. Cross connection between `saw` and `yesterday`.
![[Pasted image 20230426155057.png]]

Non projective links can't be handled with simple rules.

## Proposals to handle non-projective

Add some additional rules that skips: Actions for non-projective arcs (Attardi)

![[Pasted image 20230426155317.png]]

Example in Czech: free words

![[Pasted image 20230426155357.png]]


### Adding swap

![[Pasted image 20230426155659.png]]

Solution by Nivre

Algorithm:
 
![[Pasted image 20230426155742.png]]

## Learning

What feature we want to give to our classifier? Well some special tokens, that have an ID we can see the `F` form, `L` lemma, `P` part of speech (is more likely to connect adverb with verb), `M` morphology: can be agreement between number and words, if verb is plural the subject must be plural.

#### Training example

![[Pasted image 20230502093121.png]]


## DeSR model

![[Pasted image 20230502093416.png]]

![[Pasted image 20230502093427.png]]

Different scores

![[Pasted image 20230502094338.png]]

## Oracles problems

Oracles only suggest correct path to follow, the classifier would replicate that, but the classifier sometime makes mistakes, so an *unusual state* and the classifier would predict action on the wrong parse tree, doing this the error will propagate: mistake propagating to parser and model performs poorly, so a single error ruins all the work.

Alternative are oracles with alterative paths, so that havign more option it can recover.

## Neural Parsing

Same approach but instead using classical classifier like SVM, we use word embeddings and neural networks, this is fast and accurate, instead SVM are really slow.
Embedding learned are coupled with part of speech eg: `has_VBZ`, the parser is still a shift reduce one

![[Pasted image 20230502095054.png]]

Example of stack and buffer used by the model 

![[Pasted image 20230502095254.png]]

## Further developments

Google `Parsey McParseFace` 
![[Pasted image 20230502100049.png]]

### SOTA

Currently SoTA uses transformers, achieving 97,33% UAS paper here [Rethinking Self-Attention: An Interpretable Self-Attentive Encoder-Decoder Parser](https://khalilmrini.github.io/Label_Attention_Layer.pdf)


## Graph based parsing

![[Pasted image 20230502102619.png]]



## Biaffine attention model

Idea is to use Bidirectional LSTM over word and tag embeddings

![[Pasted image 20230502102830.png]]

### Parser 

![[Pasted image 20230502103019.png]]

### Again Self Attention applied in the model

![[Pasted image 20230502103042.png]]


## DiaParser from professor
This uses transformer to obtain representation of the input. Some attention layer happens to capture some relationships

![[Pasted image 20230502103800.png]]

```run-python

```


>[!info]
> 






---
# References

