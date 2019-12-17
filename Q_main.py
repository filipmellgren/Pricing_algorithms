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
EPSILON = PARAMS[3]
NUM_EPISODES = PARAMS[4]

# Objects
agent1 = Agent()
agent2 = Agent()

# Initializations
writer = SummaryWriter(comment="-q-iteration")
iter_no = 0

## TODOs ##
# Run  function with more natural parameterizations, like when to stop.
# When to break an episode
#   break after one billion (high number) or after convergence
# Define convergence and a target goal
# How is the observation spce updated? how dows it work?
    # maybe in value_update. 
    # Make the observation space a box so that I can add on DL
# initialize the Q table in a smarter way. P. 8 Calvano
# Calvano et al introduces a bounded memeory of k steps which defines the state
    # they use k = 1
# Also introduce their greedy policy. Then I might be able to replicate their findings
# Make one discrete environment and one continuous. Start with discrete for simplicity.
    # Replicate Calvano and make sure that results are similar. Then try new things
    # such as DQN, new demand function etc.

#### Q learning Algorithm
# loop over training episodes p.105 Sutton
for ep in range(NUM_EPISODES):
    # 1: initialise Qs
    env.reset()
    while True: # number of steps per episode
        iter_no += 1
        # 2: agents choose actions simultanously. 
        action1 = agent1.act() # does this perhaps need to take eps as arg?
        action2 = agent2.act()
        action = action1*10 + action2 # 10 hardcoded temporary. it's ncol, number of discrete actions
        # 3: outcomes are calculated
        s, reward_n, done, prob = env.step(action)        
        #s, a, r, s_next = env.interact(np.asarray([action1, action2]))
        s_next = action
        # 4: Bellman updates
        agent1.value_update(s, action1 ,reward_n[0],s_next)
        agent2.value_update(s, action2 ,reward_n[1],s_next)
        #agent1.value_update(s[0].item(), a[0].item() ,r[0].item(), s_next[0].item())
        #agent2.value_update(s[1].item(), a[1].item(), r[1].item(), s_next[1].item())
        # 5: repeat until convergence
        if iter_no > 1000:
            break
env.close()



