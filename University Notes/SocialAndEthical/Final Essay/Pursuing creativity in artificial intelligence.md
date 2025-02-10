
## Author: Vincenzo Gargano

## Matricola: 667591

# Introduction:

Creativity has long been considered a innate human trait, characterized by the idea of producing *novel* and valuable ideas, *artifacts* or performances. With the recent advancement in Artificial Intelligence, the introduction of neural net based architectures as Large Language Models (LLMs) for text and generally the families of Generative Models based on Diffusion for producing images and more advanced techniques for producing high quality music given text. At this point we can generate indistinguishable works of arts in all these different shapes by prompting arising a fundamental question about the nature of creativity itself and the potential of machines to embody it.
We will talk about several ideas drawn from Douglas Hofstadter's "Gödel, Escher, Bach: An Eternal Golden Braid" to get the basis on the concept of emergence and self-reference that appear to be the human source of that creativity. The overall foundational models are good at simulate certain creative processes. Then we will analyze philosophical and ethical implication of AI-generated content and consider future trajectories and guidelines aiming to include these new technology in our society, given the really fast excalation of their abilities (that is likely going to stop soon). 





## Chapter 1: The nature of creativity: Humans and Machines perspectives

To give some groundings to what we're going to describe, i'll try to give some formal definition and introduce some vocabulary for describing something that is not inherently formal.

```ad-Definition
Creativity is considered an interactive process where an *actor* create novel outcomes as part of different domains in varying enviroments.
```

We can find five concept at core of **creativity**: *actor* the agent that is acting and creating, a *process* the sequence of action-response of the overall "mind-state" of the actor, an *outcome* the final result of the sequence and finally a *domain* being the setting and hence the type of outcome is domain dependant. The thing i want to focus on is that the outcome has to be *novel*, but in general in the art word nothing is really original all is a mixture of real concept that are being changed slighly by each actor making something new: for example think to all eastern music based on the same chord progression that doesn't sound all the same. Also we need to consider that we humans can be influenced by our own minds and enviroments during this cognitive processes so the interplay between all those cognitive, psychological and neurological factors higlights the complexity of human creativity. Humans got also a *analogical awareness* that is what make us discover patterns without explicitly searching for them make comparisons, but can we store in a structure this analogical toughts, later we will see that with current AI methods we're "approximating" this. 

In the book Thinking Fast and Slow the author starts by defining two general system the human mind is subdivided a very fast and intuitive System 1 that is source of our biases and solves simple problems like "where to put the hands and feet while driving" and compute "2+2", and a slow System 2 that is the part of our mind that focus on difficult tasks or came into play while we learn something new. 
So we should focus our attention on the System 2 both in human and machines, one of the most interesting concept that generates complex behaviours is *recursion* or better in this case *recurrence* is at the base of both human cognition and creativity, in this context we can think to a process in which a system refers back to itself and operates on it's own output with a critic, Hofstader refers that this recurring application of local rules in our minds give birth to complex and novel ideas, so it's what we call "thinking". Hence thinking is not mechanical we cannot create an algorithm that thinks, because an algorithm is just a set of rules that is executed everytime in the same order, humans can do many reasoning steps keeping an internal state that is congruent with the past, a machine can only mimick by having a read-write memory that has almost no "plasticity" (adaptation).

LLMs are taking input, feeding back those input into themselves in a way resembling "recurrence" but it's not that really these model are just autoregressive that means that data is processed sequentially like in our minds by storing an *internal state* that is updated at each step but the way the "prediction" is made changes radically, in Auto-regression the models use the state to predict the next element (word or token) based only on the previous *elements*, instead we humans but also a certain type of neural net called Recurrent Neural Network possed a feedback loop that allows to remeber information in the same single state able to capture long range dependancies, additionally LLMs' flow of information never go back into themself but is a feedforward process, this is a big technological barrier but nonetheless we get an emergent behavior also in these type of models, in chapter 3 i'll explain better why this happens. 

Summarizing, until now we can say that in us humans creativity emerges from a non-linear complex set of recurring interactions with high level (System 2) mental processes, emerging from simpler parts inside our brain (pattern of activation of neurons for example). So the entire results is not equal to a linear sum of it's parts but a dynamical process.

Now how a machine can "mimick" these processes? We use very large scales of high quality data to train our models and by giving the model more "plasticity" so making them more bigger and more incline to change with respect to the data we few do them, these model implicitly learn the pattern and associations that we were talking about before with the *analogical awareness*. Also machines can do the exploration-exploitation of information, that to us human takes much more time, in a matter of millisecond, we can manipulate how much models can be "creative" with a mathematical parameter called *Temperature*, this parameters control the final part of the model that basically is a probability distribution across all the tokens that can be outputted, if a token has high probability will be almost certainly picked as next for prediction, so if we get high temperature the probability gets more spreaded in all the token so at the highest temperature every token will have the same probability to be picked up as next and this results in a very bad model, but with just the right level of temperature we can achieve different outputs, this is the reason why the models doesn't reply each time with the same words and certain time can reply in two different and also contradictive ways.

## Chapter 2: AI in Creative Arts Capability and Examples

The use of AI in art is not a recent development; however, its applications have significantly evolved and expanded exponentially during the last 3 years. Today, advanced AI systems are capable of creating original works of art, music, and literature, often *indistinguishable* from those produced by human artists. This chapter explores the capabilities of AI in the creative arts through various examples and case studies, highlighting both the potential and the challenges of this technology. Problems with AI generated art are especially hurting two type of artists, musicians and above all digital or not illustrators and graphic designers since they make out their living by working and creating high quality pieces of arts in different form for different purposes, for example: realizing a jingle and prepare drawing or animations for an advertisement can be expensive but nowadays the AI can generate in minutes the desired outcome with only one person prompting.

### Case study:  AI-Generated Music - The Suno and Udio Lawsuits

One prominent area where AI has made a substantial impact is in music generation. The music field is one of the most difficult to get in as a real artist and also require a lot of creativity to *emerge* from the other music and came out with something new. This is as true for humans that for AI, in fact designing these system is a very difficult technical challenge, music has a very complex structure composed of time, rythm, the harmony (how different changes in chords relate each other and make music sound interesting) then we have the instruments and their timbre, the style and genre and then the creative part: all these 'moving parts' are coupled and changing only one change the result. Music generation nowadays almost reached a level that even expert musicians can't tell if it's generated.
So AI systems, utilizing deep learning and neural networks, can compose music, replicate specific styles, and even create new genres. This capability has opened up exciting possibilities for artists and producers, allowing for new forms of collaboration and innovation. However, it also raises significant ethical and legal questions, particularly concerning copyright and the rights of original artists.

The specific case is a group of three labels: Universal Music Group (UMG), Sony Music Entertainment, and Warner Records suing two of the top names in generative AI music making, alleging the companies violated their copyright “en masse.”
The two sued comapnies use very large scaled models text-to-audio, raising succes due to large training data used, for example Udio was used to create some samples for the famous "BBL Drizzy" track that went viral and most of the people didn't even knew it was from AI when they heard it. What made come out this lawsuit was the fact that sometime the music generated from the AI has voices that were very similar to real artists, without telling directly to the model to use that voice, the suspect is that the companies scraped all the music available on the web without consent and use it to train the AI.
The lawsuits were brought by the Recording Industry Association of America (RIAA), the powerful group representing major players in the music industry, and a group of labels. The RIAA is seeking damages of up to $150,000 per work, along with other fees, this could potentially be the end for the artifical generated music, because this amount of money is likely to be impossible to pay for the companies.

Now what is happening is that the models are operating on both a "generative" and "transformative" behaviour: so most of the time the song is a new song but the timbre, or the voice or even some parts can resemble existing songs. The claim from Mikey Shulman, CEO of Suno is : "Suno is built for new music, new uses, and new musicians. We prize originality". Ceirtanly this was the aim but then some examples came out like the following: 
A song called “[Prancing Queen](https://suno.com/song/a9575656-5922-44fe-a925-b7582af7f8e4)” generated using the prompt “70s pop” contains lyrics to “Dancing Queen” by ABBA — and sounds remarkably like the band.

So the current AI still failes to be really creative 100% of the times, and what happens in all the art fields is that models could overfit: learn the training data and regurgitate it with some specific input. What can we do to solve this problem? 

## Chapter 3: Pursuing Open-Endedness 

Human have a mechanism that allow them to accumulate knowledge as a society, communicate between themselves and came out with new knowledge.
How do we embed machines with this capacity to reason and express a creative process that is composed of knowing old ideas, mixing them and then changing to create novelty? There is a fundamental property a machine needs to have to exibith creativity and in general "real intelligence".
Let's start by introducing this concept of **Open-Endedness**, this definition makes formal the aphorism of Lisa B. Soros that as observer of an *open-ended* system, "we'll be suprised but we'll be surpised in a way that makes sense in retrospect". So open-ended systems produce increasingly novel and surprising artifacts that are hard to predict, even for an observer that has learn to better predict by examining past artifacts, if a system exhibits these characteristics, so be able to produce learnable but novel artifact, then we call it open-ended. 
Foundational models are being trained on a vast amount of training data and mostly high quality data, these high quality is becoming really scarce at least in text, high quality text data is running out and this could be an hard limit to the family of LLMs.
Open-Endedness is observer-dependant, in fact learnability and novelty alone are not enough, if an AI system for example, acknowledge something as a novelty but has not the background knowledge to understand it it's not open-ended, and if the system knows it (because it's in the training data) it will not find it novel but would still be able to analyze it and understand low and high level details to then use it as it's own knowledge base fact.

Formally:

```ad-Definition
From the perspective of an observer a system is open-ended if and only if the sequence of artifact it produces is both novel and learnable.
```

So we have a system $S$ generating artifacts $x_t$ over time an observer $O$ that can has a statistical model based on the past observated artifacts, this will judge the $S$ produced artifacts and we want them to be *unpredictable* with respect to the observer of the model and *learnable* so on a long history artifacts have to be more *predictable*.

## Positive Example: AlphaGo

So a first example of system that is not general but is open-ended is AlphaGo, an AI that can beat the world champion on the game of Go, one of the most complex board game ever invented with the biggest solution space, bigger and more complex than chess. This system plays moves that would be low probability to be chosen by an human professionist observer, but which results in a winning game most of the times. Human can learn from this system and improve their win rate but AlphaGo will keep discover new policies from the past artifacts and get even better than the previous, this is self-learning. AlphaGo is so a Narrow Superhuman Intelligence this limits it's utility.

## Negative Example: LLMs

Foundation models are trained on big *fixed* dataset, if the distribution of the data is learnable the points is that these cannot be endlessly novel since the observer, humans in this case, at a certain point will have modelled epistemic uncertainty. LLMs appear to be open-ended on the broader side (since human knowledge is limited by memory) but when we go in detail in a narrow scenario such task involving planning the limitation of these models are exposed.


## Open-Endedness is an experiential process

The main limit is that AI is trained on fixed data, and OE is a continual process where novelty are produced and the observer adapts to them, we want to keep these two parts adapting each other. A first step to do this could be a self-improvement loop where model generated new knowledge in form of hypoteses, insights or creative output beyond the human knowledge and then actively engages to push the boundary to it's own capabilities. This would require a mechanism of self-evaluation, identifying what has to be improved and adapt learning accordingly. The proposals are to use Reinforcement Learning, so guide the agent trough an enviroment and use reward coming from the real world distribution, for example in the language would be to use the entire behaviour of the human coming out with new ideas. Other ideas are to use evolutionary algorithms for generation of diverse and qualitative superior variation until we get a novelty that is good enough to be considered creative by an observer. 

A different way to think to creativity is to frame it like a *exponential search problem*: we start with something that the observer can predict, thus violating novelty, but then we add steps making it more complex like searching in binary tree each time the possibilities double and it goes exponentially, humans are very good at this, finding the solution or the path that is the best one for the problem, the more is surprised the observer better it is, like we're maximizing the entropy in the temeperature parameter aforementioned that makes more creative the models.

## Safety and final considerations

Let's say we have a fully general open-ended AI sytem, this would be a Artificial Super Intelligence capable of being creative, and having general knowledge about all we humans know and it's able to self-learn and improve, this is not safe we could not predict what would happen, maybe the system would protect us because we're a data generating process out of it's distribution that can still help the system where it isn't good enough in the search space.

So the way we design the path to this type of system is fundamental for the safety concerns, also the solutions to these problems may depend on the design of the open-ended system itself. The agency of these systems poses several risks, such as goal misgeneralization, so the system misunderstands the goal and make choices that will bring to a completely different outcome, faulty reward function that can bring, again a behaviour that is the opposite of what we desire and also letting these systems search without control, for example the system could aggressively search by finding ways to get more power to search more and so on getting out of hands, there are several approaches that mitigates these type of errors in design that are being currently studied like safe exploration or humans in the loop.

So we can conclude that current AI are not creative, but we could design and aim towards more creative AI remebering we need also to address problems in the field of human creativity being harnessed by these new tools.

## Resources

1. Redefining Creativity in the Era of AI? Perspectives of Computer Scientists and New Media Artist
2. Godel, Esher, Bach (GEB) from Douglas Hofstader
3.  Thinking Fast and Slow from Daniel Kahneman
4. [Lawsuit Udio and Suno](https://www.theverge.com/2024/6/24/24184710/riaa-ai-lawsuit-suno-udio-copyright-umg-sony-warner)
5. [Open-Endedness is Essential for Artificial Superhuman Intelligence](https://arxiv.org/pdf/2406.04268)