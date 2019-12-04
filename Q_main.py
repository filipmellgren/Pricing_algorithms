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
Q0 = np.zeros((26, 10)) # Q table agent 0
Q1 = np.zeros((26, 10)) # Q table agent 1
##### Parameters
gamma = 0.99 # discount factor
alpha = 0.99 # learning rate. Should be decreasing over iterations
alpha_decay = 0.99
epsilon = 0.999 # Should also decrease
epsilon_decay = 0.9999
#reward = profit(state, action)# defined as profit in period t. Actually a table?
num_episodes = 100
num_steps = 100 # Beneficial to have a big value here when exploration is needed


#### Q learning Algorithm
# loop over training episodes p.105 Sutton
for i_episode in range(num_episodes):
    # Set initial observation
    S = env.reset()
    epsilon = 0.99
    alpha = 0.99
    # loop for each step of the episode until S is a terminal state (my case, after convergence)
    # TODO: make this a function
    for t in range(num_steps):
#?        env.render()
        # Action choice:
        if random.uniform(0, 1) < epsilon:
            A0 = env.action_space.sample() # Explore action space
            A0 = A0[0] # TODO: this is not elegant
        else:
            A0 = np.asarray([np.argmax(Q0[S[0]])]) # Exploit learned values
            
        if random.uniform(0, 1) < epsilon:
            A1 = env.action_space.sample()
            A1 = A1[0] # TODO: this is not elegant
        else:
            A1 = np.asarray([np.argmax(Q1[S[1]])])
        A = np.concatenate((A0, A1), axis=0)

        # Take action A, observe R, S_next
        S_next, R, done, info = env.step(A) # env.step(action)
        # Update Q table: TOOD: should definitely be several Q tables
        Q0[S[0], A[0]] =  Q0[S[0], A[0]] + alpha *(R[0] + gamma * np.max(Q0[S_next[0], :])- Q0[S[0], A[0]])
        Q1[S[1], A[1]] =  Q1[S[1], A[1]] + alpha *(R[1] + gamma * np.max(Q1[S_next[1], :])- Q1[S[1], A[1]])
        # If S_next is terminal: set value funciton target to R
        # Move to next stage:
        S = S_next
        epsilon = epsilon * epsilon_decay
        alpha = alpha*alpha_decay
env.close()
        
     