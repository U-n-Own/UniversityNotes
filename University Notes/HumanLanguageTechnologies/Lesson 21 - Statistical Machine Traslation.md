Date: [[2023-05-09]]

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]],[[Probability]].

# What makes good translation

When translating we want to be careful to two things *Faithfulness* and *Fluency*, so how close is to the real meaning the translated one and how is natural to read to the reader of the target language

Formalized
$$
\hat{T} = argmax_T\{\text{fluency}(T)\text{faithfulness}(T,S)\}
$$

This is the IBM model, the two thing seems familiar because they are actually probability and moreover

$$
\hat{T} = argmax_TP(T)P(S|T)
$$

Well this is Bayes!

## Fluency

How do we measure the fluency of a sentence?

**Intuition** is that: degree to which words in one sentence are plausible translations of words in other sentence. Product of probabilities that each word in target sentence would generate each word in source sentence.

## Faithfulness

Need to know for each target word probability of the mapping, how do we learn probabilities? Parallel text! We take the two text and do a comparison word by word

## Algorithm in general

**![[Pasted image 20230509100529.png]]**
## Phrase based Machine Translation

Idea is group things in phrases and then translate by pieces.

![[Pasted image 20230509100628.png]]

#TODO



>[!info]
> 






---
# References

