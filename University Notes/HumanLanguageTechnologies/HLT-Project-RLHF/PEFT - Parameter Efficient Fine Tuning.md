[[LLMs]], [[Machine Learning]], [[Fine Tuning]].

## Motivation

LLMs are getting bigger and more bigger they are better are the performance, but fine tune them becomes easily very costful so techniques are required to make more efficient this step, having a Fine Tuned model is crucial to a lot of task like [[RHLF]] of LLMs or simply customizatio wrt certain subtaska we want a LLM to be more good at.

Also we have models that go out from language so [[ViT]]...

## Idea:

So basically we want to reduce the parameters to train our model when doing FT, we freeze the model by only training a very little space or parameter added as extra, decreasing the cost and overcoming the problem of [[Catastrophic Forgetting]], a behaviour oberved when doing finetuning of these models, we got also portability because we get tiny checkpoints (for example a FT model can take 40GB if we finetune it, but with PEFT only few MB...)

## Methods

So the main methods use for doing PEFT is [[QLoRA & LoRA - Low Ranked Adaptation & Quantization]] 