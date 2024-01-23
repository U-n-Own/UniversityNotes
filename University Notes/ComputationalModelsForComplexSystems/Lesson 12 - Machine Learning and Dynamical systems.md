Date: [[2023-04-27]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]],[[Computational Biology]][[Dynamical and Complex Systems]][[Machine Learning]],[[Graph Neural Networks]],[[]]

# Paper presentation: Prediction of Dynamical Properties of Biochemical Pathways with Graph Neural Networks

Two groups working on this:

From BioSystems modelling group @UniPi, study properties of biological system with background in formal methods, using the same tools to study properties of formal languages.

CIML group @UniPi from Alessio Micheli, Bacciu, etc...

## Why?

Many reaction inside cells are better described by graphs

![[Pasted image 20230427113237.png]]

We would like to study how concentration of some molecules in a certain enviroment changes, so dynamics over time, depends on these graphs and if you change graph you obtain different dynamics. There is a correletion between graph structure and dynamics, the aim is to train networks on graphs to learn this structure. A cell is a very small *complex system*.

Every function inside the cell involves a lot of thing: proteins, structures et cetera eg: metabolism.

There are a lot of pathways:

- Metabolic pathways are a series of chemical reactions that occur within a cell to break down or synthesize molecules. These pathways are essential for maintaining the energyand materials needed for cell growth, reproduction, and function.

- One example of a metabolic pathway is cellular respiration, which involves the breakdown of glucose. 
- Signalling pathways are the mechanisms by which cells communicate with each other to coordinate their functions and response to external stimuli. These pathways involve a complex network of molecules that interact with each other to transmit signals from the cell surface to the nucleus, where they can alter gene expression and ultimately affect cellular behavior.
- Gene regulatory networks are complex networks of genes and their regulatory interactions that control the expression of genes in a cell or organism. These networks play a critical role in many biological processes such as development, differentiation, and response to environmental stimuli.

We can find different graph notation, this is the EGF Pathway, usually these signal are proliferation signal, we a cell get the signal there is a chain reaction where the interaction are activated activating proteins that are responsible for cell growth for example.
![[Pasted image 20230427114015.png]]

### Biochemical pathways in SBML : (System Biology Markup Language)

![[Pasted image 20230427114257.png]]

So a pathways is a list of reactions each with lists of reactants, products and modifiers
There exists database of pathways.

### Simulation of pathways dynamics

![[Pasted image 20230427114618.png]]

### Dynamical properties

Sensitivity is how much the results change when we change a concentration of molecule.

Robustness instead is like how the system try to remain stable if we do big changes, to do this we need to do big number of simulations, so we would like to do less simulation.

![[Pasted image 20230427114719.png]]

## Idea:

Try to find similarities and correlation between graph based system and dynamical properties of the systems.

![[Pasted image 20230427115010.png]]

### Approach:

Take pathways simulate in order to generate a dataset that has information about dynamical properties, after this taking a lot of time we have some information, and we can train ML models to this relationship between model and graphical properties

![[Pasted image 20230427115123.png]]

The research was focussed on *Concentration of Robustness*.

The network is robust if we vary initial concentration of "A" by 20% if the change in steady concentration of other species is less than 20%, this means that the others are not affected by these changes. This is called $\alpha$-Robustness as the ratio of size of steady state concentration of external state and initial concentration of the specie we re analyzing.

![[Pasted image 20230427115338.png]]

### Methodology

After dataset is generated : 7k Graphs with labelled robustness, so taking a couple of input-output for each molecule.



![[Pasted image 20230427115821.png]]

### Graphs preprocessing

Before giving the graphs to the ML model there is some preprocessing, removing some information so feeding a subgraph, removing all numeric parmeters (Kinetic Constants) these are useful for simulation but in training we want to learn a topology and infer information, also we can change kinetic constant in a reaction that has the same ratio, another reason was that parameters are difficult to estimate, so learning from the graph without parameters can generalize better to other graphs. So *Pruned quantitative informations*.
Then we pruned some part of the graphs that do not interact the things we're analyzing, for example if we want to study the subgraph containing A,B we take just the subgraph we can see in the image at right up, where only A,B are present, different would be if we want to study C and F.

![[Pasted image 20230427120504.png]]

They started with subgraph of 40 nodes

![[Pasted image 20230427120715.png]]

## Machine Learning : GNN

![[Pasted image 20230427121041.png]]

The thing is to find a **Node embedding** and **neighborhood aggregation** for a given graph.

Node embedding is the process of representing each node in a graph as a vector or point in a high-dimensional space. The goalof node embedding is to capture the structural information of the graph in a way that can be easily used for downstream tasks such as node classification, link prediction, and graph visualization. Node embedding algorithms aim to learn representations that preserve the proximity of nodes in the original graph, meaning that nodes that are close toeach other in the graph should also be close to each other in the embedding space. 

![[Pasted image 20230427121211.png]]


![[Pasted image 20230427121458.png]]

Note that big graphs are predicted better than smaller graphs, in fact in smaller graphs the kinetic parameter have big impact on the results, in big graphs this thing tends to be ininfulent

![[Pasted image 20230427121606.png]]


>[!info]
> 






---
# References

