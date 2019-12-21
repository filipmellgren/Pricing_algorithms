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
from prisoner_dilemma import PrisonerDilemma
from config import avg_profit_gain
# Parameters
env = PrisonerDilemma()

from config import PARAMS
GAMMA = PARAMS[0]
ALPHA = PARAMS[1]
BETA = PARAMS[2]
NUM_EPISODES = PARAMS[3].astype(int)
nA = PARAMS[5].astype(int)
ITER_BREAK = PARAMS[7].astype(int)
CONV = PARAMS[8].astype(int)
# Objects
agent1 = Agent()
agent2 = Agent()

# Initializations
writer = SummaryWriter(comment="-q-iteration")
iter_no = 0

## TODOs ##
# Replicate Calvano:
    # Add better values for price range and profits
    
# Discrete case
    # Add own specification

# Continuous case
    # Make the observation space a box so that I can add on DL
   
# Other
    # Try new things: 
        # such as DQN, new demand function etc.

#### Q learning Algorithm
# loop over training episodes p.105 Sutton
profits = np.zeros((ITER_BREAK+2,NUM_EPISODES+2))

for ep in range(NUM_EPISODES): # Start out with fewer episodes
    print(ep)
    # 1: initialise Qs
    env.reset() # calls discrete.Discrete.reset() is this what I want?
    agent1.reset()
    agent2.reset()
    iter_no = 0
    s_next = 0
    while True: # number of steps per episode
        iter_no += 1
        eps = 1 - np.exp(-BETA * (iter_no))
        # 2: agents choose actions simultanously. 
        action1 = agent1.act(eps)
        action2 = agent2.act(eps)
        action = action1*nA + action2 # converts two actions into an interpretable action
        # 3: outcomes are calculated
        s = s_next # remember old state
        s_next, reward_n, done, prob = env.step(action)
        # 4: Bellman updates
        agent1.value_update(s, action1 ,reward_n[0],s_next)
        agent2.value_update(s, action2 ,reward_n[1],s_next)
        # 5: repeat until convergence
        # Store profit
        profits[iter_no][ep] = reward_n[0]
        # Check if optimal action has stayed constant for 25k repetitions
        if iter_no > ITER_BREAK or agent1.length_opt_act > CONV:
            if agent1.length_opt_act > CONV:
                print("yay")
                print(iter_no)
            break

env.close()

# TODO: calculate mean profit gain over episodes
import pandas as pd
x = profits.mean(axis = 1)
y = profits[1000:1900000]
z = pd.DataFrame(y)
z = z.iloc[:, [0,1,2, 3,4]] 
w = np.mean(z, axis =1)
w2 = w.rolling(1000, min_periods = 1).mean()
z2 = z.rolling(1000).mean()
z3 = avg_profit_gain(z2)
w3 = avg_profit_gain(w2)
plt.plot(w3)



