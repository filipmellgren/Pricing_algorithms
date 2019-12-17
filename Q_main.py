#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:54:30 2019

@author: Filip Mellgren

!git add "Q_main.py"
!git add "agents.py"
!git add "bertrand_nash.py"
!git add "prisoner_dilemma.py"
!git add "config.py"
!git commit -m "My_commit"
!git push origin master

Two Q-tables are necessary, one for each competitor
This makes the environment "non stationary" – i.e. hard to solve and not guaranteed to converge
Approach taken: so called "independent learning". Agents don't necessarily know they are competing, they simply observe profits for various prices
Need Deep learning to approximate neighbouring cells
Important difference between this and Calvano's paper is what's observable. 
     In this case, players observe their own profit and never the opponents price.
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import gym
import collections
from tensorboardX import SummaryWriter
from agents import Agent  

# Parameters
from config import PARAMS
env = PARAMS[0]
GAMMA = PARAMS[1]
ALPHA = PARAMS[2]
BETA = PARAMS[3]
NUM_EPISODES = PARAMS[4]

# Objects
agent1 = Agent()
agent2 = Agent()

# Initializations
writer = SummaryWriter(comment="-q-iteration")
iter_no = 0

## TODOs ##
# Replicate Calvano:
    # update explore exploit schedule
    # Update when to break an episode (define convergence)
    # Initialize the Q table in a smarter way. P. 8 Calvano 

# Continuous case
    # Make the observation space a box so that I can add on DL
   
# Other
    # Calvano et al introduces a bounded memeory of k steps which defines the state
        # they use k = 1
    # Try new things: 
        # such as DQN, new demand function etc.

#### Q learning Algorithm
# loop over training episodes p.105 Sutton
rew = np.zeros((100002,NUM_EPISODES))
for ep in range(NUM_EPISODES):
    # 1: initialise Qs
    env.reset()
    iter_no = 0
    s_next = 0
    while True: # number of steps per episode
        iter_no += 1
        eps = 1 - np.exp(-BETA * (iter_no))
        # 2: agents choose actions simultanously. 
        action1 = agent1.act(eps)
        action2 = agent2.act(eps)
        action = action1*10 + action2 # 10 hardcoded temporary. it's ncol, number of discrete actions
        # 3: outcomes are calculated
        s = s_next # remember old state
        s_next, reward_n, done, prob = env.step(action)        
        #s, a, r, s_next = env.interact(np.asarray([action1, action2]))
        # 4: Bellman updates
        agent1.value_update(s, action1 ,reward_n[0],s_next)
        agent2.value_update(s, action2 ,reward_n[1],s_next)
        #agent1.value_update(s[0].item(), a[0].item() ,r[0].item(), s_next[0].item())
        #agent2.value_update(s[1].item(), a[1].item(), r[1].item(), s_next[1].item())
        # 5: repeat until convergence
        rew[iter_no][ep] = reward_n[0]
        if iter_no > 100000: # Calvano takes it up to a billion or until convergence!
            break


env.close()

