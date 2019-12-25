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

!git add "Q_main2.py"
!git add "agents2.py"
!git add "discr_bertrand.py"
!git add "config2.py"
!git commit -m "commit"
!git push origin master

!git pull

From the correct directory in the terminal write:
    !tensorboard --logdir runs --host localhost
Then go to:
    http://localhost:6006/
in the browser

"""
# TODO
    # config2: Max and min profits are incorrect
    # discr_bertrand: Necessary to keep the state numeration?
    # config2: derive environment boundaries analytically
    # Pilot test
    # Pass variables as arguments in functions for speed and clarity
    # Raise exceptions early
    # functools to set cache size
    # linked lists instead of arrays
    # Mind that ging local from global costs memory while it saves speed


# Libraries
import numpy as np
import matplotlib.pyplot as plt
import gym
import collections
from tensorboardX import SummaryWriter
from agents2 import Agent
from discr_bertrand import DiscrBertrand
from config2 import avg_profit_gain
from config2 import rew_to_int
from config2 import to_s
import time

# Parameters
env = DiscrBertrand()

from config2 import PARAMS
GAMMA = PARAMS[0]
ALPHA = PARAMS[1]
BETA = PARAMS[2]
NUM_EPISODES = PARAMS[3].astype(int)
nA = PARAMS[5].astype(int)
nS = PARAMS[6].astype(int)
ITER_BREAK = PARAMS[7].astype(int)
CONV = PARAMS[8].astype(int)

from config2 import ECON_PARAMS
C = ECON_PARAMS[0]
A = ECON_PARAMS[1]
A0 = ECON_PARAMS[2]
MU = ECON_PARAMS[3]
MIN_PRICE = ECON_PARAMS[4]
price_range = ECON_PARAMS[5]

# Objects
agent1 = Agent()
agent2 = Agent()


# Initializations
writer = SummaryWriter(comment="-q-iteration")
iter_no = 0


# Q learning Algorithm

start = time.time()

for ep in range(NUM_EPISODES): 
    print(ep)
    # 1: initialise Qs
    env.reset()
    agent1.reset(nS, nA, GAMMA, C, A, A, A0, MU, price_range, MIN_PRICE)
    agent2.reset(nS, nA, GAMMA, C, A, A, A0, MU, price_range, MIN_PRICE)
    iter_no = 0
    s_next = 0
    while iter_no < ITER_BREAK and agent1.length_opt_act < CONV:
        iter_no += 1
        eps = 1 - np.exp(-BETA * (iter_no))
        # 2: agents choose actions simultanously. 
        action1 = agent1.act(eps)
        action2 = agent2.act(eps)
        action = action1*nA + action2
        # 3: outcomes are calculated
        s = s_next
        s_next, reward_n, done, prob = env.step(action)
        # 3.1: Actual state of the environment is not observed by the agents.
        reward1 = rew_to_int(reward_n[0])
        reward2 = rew_to_int(reward_n[1])
        s_next1 = to_s(action1, reward1)
        s_next2 = to_s(action2, reward2)
        # 3.2 Add to writer (add both using add_embedding?)
        writer.add_scalar(str(ep), reward1, iter_no)
        # 4: Bellman updates
        agent1.value_update(s, action1 ,reward_n[0],s_next1, ALPHA, GAMMA)
        agent2.value_update(s, action2 ,reward_n[1],s_next2, ALPHA, GAMMA)
        # 5: repeat until convergence
    if agent1.length_opt_act > CONV:
        print(iter_no)

writer.close()
env.close()

end = time.time()
print(end - start)

# Before: 155.4 seconds
# After changing while statement: 154.5 seconds (w00p)
# After changing from globals to locals in agents2: 153:8
# 154.83, 153.3
