#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 10:46:51 2019

@author: Filip Mellgren

This is the main file for a multi agent reinforcement learning application
where two agents compete in a discrete Bertrand environment, i.e. having price
as their strategic variable. The agents are able to see their own prices and
profits K periods back. They are unable to see each others prices and profits.

Question is, can they still learn how to cooperate?

!git add "Q_main.py"
!git add "agents.py"
!git add "discr_bertrand.py"
!git add "config.py"
!git commit -m "tmp"
!git push origin master

!git pull
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import gym
import collections
from tensorboardX import SummaryWriter
from agents import Agent
from discr_bertrand import DiscrBertrand
from config import avg_profit_gain

# Parameters
env = DiscrBertrand()

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


# Q learning Algorithm
profits = np.zeros((ITER_BREAK+2,NUM_EPISODES+2))

for ep in range(NUM_EPISODES):
    print(ep)
    # 1: initialise Qs
    env.reset()
    agent1.reset()
    agent2.reset()
    iter_no = 0
    s_next = 0
    while True:
        iter_no += 1
        eps = 1 - np.exp(-BETA * (iter_no))
        # 2: agents choose actions simultanously. 
        action1 = agent1.act(eps)
        action2 = agent2.act(eps)
        action = action1*nA + action2
        # 3: outcomes are calculated
        s = s_next
        s_next, reward_n, done, prob = env.step(action)
        # 4: Bellman updates
        agent1.value_update(s, action1 ,reward_n[0],s_next)
        agent2.value_update(s, action2 ,reward_n[1],s_next)
        profits[iter_no][ep] = reward_n[0]
        # 5: repeat until convergence
        if iter_no > ITER_BREAK or agent1.length_opt_act > CONV:
            if agent1.length_opt_act > CONV:
                print("yay")
                print(iter_no)
            break

env.close()