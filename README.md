# Pricing_algorithms
What the project does:

This project aims to explore whether multiagent reinforcement learning (MARL) algorithms can learn to collude in an economic environment with partial observability (a so called "POMDP").

First, I replicate an existing paper inside "repl_calvano" in the fodler repl_calvano (link to paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3304991). This paper establishes that MARL can indeed learn how to collude and serves as a benchmark for me to test that my code works. I.e. once I've replicated it I'm confident that the code is correct and can be taekn to the next step.

Once the code is able to replicate (at a rough level), I stick to the discrete environment and implement changes to the economic environment that are meaningful from an economic perspective. This is where a novel contribution of mine enters and the results will be presented in my MSc thesis at SSE/MA thesis at St. Gallen University. This is done in the folder disc_part_obs.

Finally, if time permits, I want to take the development one step further and make the observation space and action space continuous. This will require function approximation via deep learning (which are new techniques to me). This will allow me to explore a richer set of environments and make a larger contribution to the field.

Why the project is useful:

It contributes to the industrial organization litterature on tacit collusion in a digital age.

Some additional notes on structure:
All folders contains files for:
* The main logic. This has a name similar to Q_main.py (might come with an enumeration) and is where the core loop happens and the Q values are being identified.
* Configuarations. In a file named similar to config.py, parameters are defined by the user. Parameters and functions form this file are called by the other files. In this way, I only ever have to define these values, that are common, once.
* The agents. These are called agents.py and include a class that defines the agents.
* An environment. Names differ depending on the environment. This describes the overall environment where the agents compete and the transition dynamics present in the environment.
