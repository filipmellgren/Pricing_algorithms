#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:54:30 2019

@author: filip


!git add "Q_main.py"
!git add "agents.py"
!git add "bertrand_nash.py"
!git add "bertrand_nash.py"
!git commit -m "My_commit"
!git push origin master
"""
## Libraries
import numpy as np
import random
import matplotlib.pyplot as plt
#import environment as env
from bertrand_nash import BertrandNash
import gym
import collections
from tensorboardX import SummaryWriter
from agents import Agent
from config import PARAMS
#### Load environment and Q table structure
# Two Q-tables are necessary, one for each competitor
# This makes the environment "non stationary" – i.e. hard to solve and not guaranteed to converge
# Approach taken: so called "independent learning". Agents don't necessarily know they are competing, they simply observe profits for various prices
# Need Deep learning to approximate neighbouring cells
# Important difference between this and Calvano's paper is what's observable. 
    # In this case, players observe their own profit and never the opponents price.
    
# TODO: initialize the Q table in a smarter way. P. 8 Calvano
# Parameters

env = PARAMS[0]
GAMMA = PARAMS[1]
ALPHA = PARAMS[2]
EPSILON = PARAMS[3]


# Objects
agent1 = Agent()
agent2 = Agent()

# Initializations
writer = SummaryWriter(comment="-q-iteration")
iter_no = 0

## TODOs ##
# Run  function with more natural parameterizations, like when to stop.
# Number of episodes, when to break an episode
# Define convergence and a target goal
# How is the observation spce updated?
    # maybe in value_update. 
    # May want to start out simple and define discrete outcomes in the demand function
    # And then make the observation space a box when I add on DL

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
        # 3: outcomes are calculated
        s, a, r, s_next = env.interact(np.asarray([action1, action2]))
        # 4: Bellman updates
        agent1.value_update(s[0].item(), a[0].item() ,r[0].item(), s_next[0].item())
        agent2.value_update(s[1].item(), a[1].item(), r[1].item(), s_next[1].item())
        # 5: repeat until convergence
        if iter_no > 100:
            break
env.close()


