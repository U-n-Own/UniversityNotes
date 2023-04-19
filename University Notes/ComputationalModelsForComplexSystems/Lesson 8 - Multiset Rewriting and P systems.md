Date: [[2023-04-06]]

Status: #notes

Tags: [[Complex Systems]], [[A.I. Master Degree @Unipi]]

# Formal Language Theory intro

Today we will start second part of the course, languages can be based on rules and those based on particular graphs like petri nets, language coming from theory like automata or grammars, for example automata checks for Soundness instead Grammars are capable to produce our "theorems"

## FLT representation

rule based: grammars
automata based regular automata
algebraic: rel exp

| Formal Language  | Description |
|---|---|
| Regular Language  | A type of formal language that can be represented by a regular expression or a finite automaton. It is used to describe patterns in strings such as email addresses or phone numbers.  |
| Context-Free Language  | A type of formal language that can be represented by a context-free grammar. It is used to describe patterns in nested structures such as programming languages or markup languages like HTML. |
| Context-Sensitive Language  | A type of formal language that can be represented by a context-sensitive grammar. It is used to describe patterns in more complex structures where the rules for generating the language depend on the context of the symbols being generated. |
| Recursive Language | A type of formal language that can be defined in terms of itself, either directly or indirectly. It is used to describe patterns in self-referential systems such as fractals or recursive functions in computer science. |

Note: There are other types of formal languages beyond these four (e.g., Chomsky hierarchy), but these are some of the most common and well-known ones.

## Concurrency Theory


|                | FLT (real lang)  | Concurrency theory         |
| -------------- | ---------------- | -------------------------- |
| Rule based     | Grammars S->ab   | $P_1 + P_2 -> P_1' + P_2'$ |
| Automata based | regular automata |                            |
| Algebraic      | Rel exp `a*b`    |                            |

A Petri net is a mathematical model used to describe concurrent systems. It consists of two types of nodes: places and transitions. Places represent states or conditions of the system, while transitions represent events or actions that can occur in the system. These nodes are connected by directed arcs, which indicate that some resource or token can move from one place to another through a transition.

Petri nets are commonly used to model and analyze complex systems in various fields such as computer science, engineering, and biology. They can help identify potential problems in a system's design, optimize its performance, and ensure its correctness.

In the context of rule-based grammars such as S->ab, a Petri net could be used to model the possible sequences of rules that can be applied to generate valid strings based on the grammar rules. Each place would represent a state of the string being generated (e.g., "S" for the initial rule), while each transition would represent an application of a grammar rule (e.g., "S -> ab"). The arcs would indicate how tokens (symbols) move between these states through the application of these rules.

We can describe parallel composition like this: $a.P_1' | \overline{a}.P_2'$ where the pipe sign say that the two are parallel.

### Multisets for chemical reactions

A multiset is a variant of mathematical sets that permits duplicates
![[Pasted image 20230406114327.png]]

Given a support set $\Sigma$, the mathematical representation of *multiset* M over $\Sigma$ is
- A **set of pairs** in $M \subseteq \Sigma \times \mathbb{N}$

We can even represent multisets as *strings*, the Kleene closure is the *Set of all possible multisets* called $\Sigma^*$

#### Multisets union is concatenation

![[Pasted image 20230406114916.png]]

## Rewriting systems and multisets

![[Pasted image 20230406120022.png]]

And we can define a set of inference rules so doing this defining a system of rewriting

![[Pasted image 20230406120105.png]]

### Rewriting systems seen ad transition systems

![[Pasted image 20230406120346.png]]

Further we can see these as by adding stochastic rates! So we're building Continuous Time Markov Chains.
![[Pasted image 20230406120439.png]]

#### Semantics with stochastic MSR

Now definition changes because we're adding to our system a new rule, that adds the stochastic rate.
![[Pasted image 20230406120519.png]]

## Exploiting parallelism

![[Pasted image 20230406121919.png]]

### Semantic of simple and maximal parallelisms

The first definition of paralel semantic (simple) can apply 1 or more steps, for example the last rules says that we can do the two path togheter instead we in the first can apply the inference rule to change only one and leave the other unchanged.

![[Pasted image 20230406122000.png]]
 
An example we can do is with following rewriting $R_1: AB \mapsto C, R_2: A \mapsto B$

$A^3B^2 \mapsto A^1B^2C$ by applying the first $R_1$ then $R_2$. 

Maximally parallel instead used all the possible rules until it reaches the normal form the transition in the maximal cases are subsets of the cases in the simple parallelism, this because the simple parallel semantics is more restrictive and the maximal one let use relax and get a system more powerful.


## Multisets in formal language theory

Ignoring sequential ordering. Make a broader language 
![[Pasted image 20230406123301.png]]


### Maximal parallelism make system more expressive

![[Pasted image 20230406123536.png]]

## P systems

Biologically inspires system, like membranes of cell they have hierarchical structure, each of these structure is the **membrane** each membrane has it's multisets and rules, we can compute function at different levels

![[Pasted image 20230406123733.png]]

Example of P system

We can see the hierarchy of membranes with rules associated, this is used to model distributed computations. We can move sumble by one membrane to another by using `out` and `in` tokens, like the rule we're seeing the $ab \mapsto dd_{out}e_{in}$
![[Pasted image 20230406123911.png]]

### Formally defining P systems

![[Pasted image 20230406124208.png]]

Evolution rules: Non competitive one can be seen as context free and context sensitive the cooperative one
We have both of one in this systems

![[Pasted image 20230406124730.png]]

## Exploiting P systems to compute functions

We give $n$ number of $a$ that are being processed we can get two object by the consumption $Y,N$, if we get the first then is even otherwise is odd, we can see that this system is Turing Complete
![[Pasted image 20230406124500.png]]

We can use `Big-Step semantics` aka Maximal parallelism

![[Pasted image 20230406125042.png]]

#### P systems describing $n^2$

Well we're just applying rules to the $a$, then sequentially we transform our strings at the step 3 $b_2 \mapsto b_2e$ we have $n^2$ copies of $e$, the last rule is the dissolving rule that "dissolves the first membrane" sending them into 1. If happens that computations dissolve after less step than we wanted then we get some $b_1$ that we don't want, how do we do? We add a rule that signal an error $b_1 \mapsto b_1$. So only terminated computation are actually correct.
![[Pasted image 20230406125328.png]]

## P Systems are Turing Complete

The idea is that we can use an exponential size of membrane at each steps to make the problem in class NP  be computed in polynomial time.
![[Pasted image 20230406125859.png]]
![[Pasted image 20230406130022.png]]

## P systems models biological system

![[Pasted image 20230406130101.png]]

#### Example of biological P system
![[Pasted image 20230406130137.png]]





>[!info]
> 






---
# References

