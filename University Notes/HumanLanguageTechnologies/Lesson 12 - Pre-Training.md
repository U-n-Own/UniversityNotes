Date: @today

Status: #notes

Tags: #hlt[[A.I. Master Degree @Unipi]]

# Pretrain whole model

As said ealier pre training transformers (LLM) can be useful to do a lot of things, this technique create a model capable to do a lot of thing in a generalized way, bigger the model better the results, but also we can use the pre-training step to make further and complex model without starting from scratch everytime, other than this we can finetune the model to a specific task.

## GPT

In 2018 there was a big succes in pretraining decoder, this had 12 layers, 768 hidden state and 3072 feedforward hidden layers, trained on a corpus of over 7000 books.

But how do we give the input to our decoder for finetuning task?

Natural Language Inference: label pairs of sentences as entailing, contradictory or neutral: 
- Premise : The man is in the doorway
- Hypotesis: the person is near the door

## Bert 

We want an entailment of this: so we simply put into the encoder one after another after a special token `[DELIM]` and at the end we use a special token `[EXTRACT]` to get prediction on this entailment, the classifier take the output of extract token (dummy token). The hidden representation of that is given to the decision make to understand if there is a connection or not.

If we have three choices we can put then in the same context input in different instance of transformer and then pick the higher probability.

### Encoders

...


### Masked LM

We want to mask our $k\%$ of the input words, and predict the masked words, for example we mask some of the words and replace them with special token `[MASK]`, if we use a lot of masking context is lacking and model is very poor, if we use little making it becomes too expensive to train, masking is set to $15\%$ usually. If there are masked token that are never seen? Solution is to put a random word instead, so 80% we put the masking, 10% we use random words and 10% of the time these are kept like that.
This is just a generalization of word to vec here we give the whole sentences, we're not limited to the lenght context isn't lost over long sequences, and then wordvectorize and predict the word in arbitrary way.

### Next sentence prediction

Another way Bert is trained is to learn *relationships*, so given two sentences A,B we want to know if B follows from A (logically).

For example.

- A: The man went to the store
- B: He bought milk
- Label: ItFollows

Negative example

- A: The man went to the store
- B: Penguins fly into the sky
- Label: DoenstFollow

### Word Pieces

Bert use a variant of wordpieces model, we don't want a vocabulary to be very similar, a lot of words have suffixes and prefixes, we break those words in pieces like $hypatia$ = $h\#\#yp\#\#ati\#\#a$, word less common are split into pieces, istead common words are treated like unique tokens.

So when training transformers we will use a tokenizer to create a wordpiece representation

>[!info]
> 






---
# References

