#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:54:30 2019

@author: filip


!git add "Q_main.py"
!git commit -m "My commit"
!git push origin master

"""
## Libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import math
#import environment as env
from bertrand_nash import BertrandNash
import gym
import random

#### Load environment and Q table structure
env = BertrandNash() # read environment into this
# env.observation_space.n,env.action_space.n
#Q = np.zeros((env.state_size, env.action_size)) # Q table
Q = np.zeros((10, 10)) # Q table
##### Parameters
gamma = 0.99 # discount factor
alpha = 0.1 # learning rate. Should be decreasing over iterations
epsilon = 0.85 # Should also decrease
epsilon_decay = 0.98
#reward = profit(state, action)# defined as profit in period t. Actually a table?
num_episodes = 100
num_steps = 10000 # Problem with larher values is S_next


#### Q learning Algorithm
# loop over training episodes p.105 Sutton
for i_episode in range(num_episodes):
    # Set initial observation
    S = env.reset()
    epsilon = 100
    # loop for each step of the episode until S is a terminal state (my case, after convergence)
    for t in range(num_steps):
#        env.render()
        # Action choice:
        if random.uniform(0, 1) < epsilon:
            A = env.action_space.sample() # Explore action space
        else:
            A = np.argmax(Q[S]) # Exploit learned values
        A = A.item()
        # Take action A, observe R, S_next
        S_next, R, done, info = env.step(A) # env.step(action)
        # Update Q table:
        Q[S, A] =  Q[S, A] + alpha *(R + gamma * np.max(Q[S_next, :])- Q[S, A])
        # If S_next is terminal: set value funciton target to R
        # Move to next stage:
        S = S_next
        epsilon = epsilon * epsilon_decay
env.close()
        
      
Q = np.zeros((2, 4))