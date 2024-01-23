All from this bible: [[Artificial Intelligence - A Modern Approach (3rd Edition).pdf]]
## Part 0: Intelligent Agents 

A discussion about type of agent, enviroments and a lot more treated in the book.

**Agents** perceive the **enviroment**, it acts based on some **performance measures** in order to maximize this (rational agent), for example in RL we have the *Reward signal*, or in other cases we want to minimize a cost of certain action, the enviroment is the space in which the agent can do **actions**, doing those usually change the enviroment and the **state** in which the agent. 

Enviroment can be fully observable or partially or unobervable, can be stochastic (game with dice or real word) or can be deterministic, more than one agent can be inside our enviroment and the sequence of action can be episodic or continue, dynamic or static, discrete or continuous.

Agent can decide what to do in different ways we have model-based decision making agent that take action with respect of the percepts and can be Goal-based  or utility-based so take action with the respect of a utility function or a goal to achieve. 

Agent improve performance by learning.


Agent obey the

## Part 1: Search (Informed, Uninformed) 

Definition of a problem: initial state, actions, transition model, goal test, path cost, state space and a **path** or solution that is returned.

A review on some algorithm like A*, BFS, DFS, Greedy, Bidirectional...etc: their good points and bad points.
Memory is key sometimes: *algorithm that do not know their story are doomed to reapeat it*.

Why A* is complete? 

On the last part we talked about relaxation of problems that make them more approachable by getting good heuristics, these can be learned from experience (RL) or even computed by algorithm
## Part 2: Beyond classical search

In this chapter we moved from a simple setting to a more refined and difficult one where we dropped the determinism of the enviroment sometimes (as requirement), we saw **Hill climbing** as a very simple technique, greedy that essentially take only 1 node in memory at a time and pick the best current node, this of course suffers of some problem, not enough exploration and so fell in local minimum/maxima, then we enhanced it with stochasticity and got **Simulated annealing**, where we allow more exploration at the start and then gradually take only the best solution, so we are guaranteed to reach the global maximum if we have an appropiate schedule (temperature over time so a function from time to temperature).

Then a little glance to convex optimization and **Genetic Algorithms**: where we treat states and actions as a population so we're doing a stochastic hill climbing and as the population becomes better and better the core of this algorithm the *crossover* (action of taking two member of the population and split their information to get a new member child) becomes less and less random because of the population being all the same, also we add mutation to the child so that we ensure more exploration is done.

For non deterministic enviroment we can use AND-OR search so a DFS recursive that use contigency plans, when the enviroment is partially observable we can build a belief state representing the set of all the possible state an agent might be in, and for better exploration where we have no idea abut state and action pair, given safely explorable (no moves that are irreversible) we can go for an online search to build a map and find a goal to explore and exploit.

## Part 3 : Adversarial Search


## Part 4: Logical Agents



### Forward vs Backward chaining algorithms for entailing sentences from KB

Backward chaining is goal based when speaking of inference engines, while forward chaining is data driven (start from facts and then progressively adds information in KB), so we can say that a forward chaining is like a search without objective tyring to reach something using all the possible path, instead backward (exploit knowledge) is only using the stricly necessary information in KB (facts) to entail the query

