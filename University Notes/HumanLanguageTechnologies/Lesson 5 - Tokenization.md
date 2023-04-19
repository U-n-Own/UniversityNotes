Date: [[2023-03-14]]

Status: #notes

Tags: #hlt, [[A.I. Master Degree @Unipi]], [[Human Language Technologies]]

# Tokens

When we want to feed some text to our classifier we have to distinguish how the words are distributed in the corpora, but this isn't simple, there are certain languages that make this very difficulty, like German with some long words with multiple meaning that represent a single concept, or Chinese or Japanese, but also english, for example we can ask if "isn't" is a token or is composed by both "is+not" so it's 2 tokens in 1 or maybe 2 distinct.

## Lemmatization

Another problem is when more words have similar but not identical meaning, this can change the context sometime
![[Pasted image 20230314094811.png]]
![[Pasted image 20230314095100.png]]

## Normalization

We need to normalize terms, so if we mean "Window", we can get "window, windows, Window and Windows" as results.

**Morphology** is another important thing, *morphemes* are small units of words that have a meaning

>[!info]
> 






---
# References

