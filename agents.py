#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:08:07 2019

@author: filip
"""

import gym
import collections
import numpy as np
from tensorboardX import SummaryWriter
from bertrand_nash import BertrandNash
#from __main__ import PARAMS
from config import PARAMS

env = PARAMS[0]
GAMMA = PARAMS[1]
ALPHA = PARAMS[2]
EPSILON = PARAMS[3]

class Agent:
    def __init__(self):
        #self.env = gym.make(ENV_NAME)
        self.env = env
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)
    
    def act(self):
        # Make this depend on some epsilon greedy policy or Boltzman annealing
        if np.random.uniform() > EPSILON:
            _, action = self.max_value_action()
           # action /= 10 # Wonder if this is how to get the actions in range?
        else:
            action = self.env.action_space.sample()
        
        return action
    
    def max_value_action(self):
        # works a bit like argmax
        # "max_value_argmax_action"
        max_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value  = self.values[(self.state, action)]
            if max_value is None or max_value < action_value:
                max_value = action_value
                best_action = action
        return max_value, best_action
    
    def value_update(self, s, a, r, next_s):
        # Update states here?
        best_v, _ = self.max_value_action()
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-ALPHA) + new_val * ALPHA
        self.state = next_s

    def play_episode(self, env):
        '''
        play_episode() plays the episode with belief of optimal actions
        doesn't affect anything, it only evaluates current policy
        '''
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward # Add avg profit gain here?
            if is_done:
                break
            state = new_state
        return total_reward
