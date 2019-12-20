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
from prisoner_dilemma import PrisonerDilemma
#from __main__ import PARAMS
from config import PARAMS
from config import profit_n

env = PrisonerDilemma()
GAMMA = PARAMS[0]
ALPHA = PARAMS[1]
nA = PARAMS[5]
nS = PARAMS[6]
nA = nA.astype(int)
nS = nS.astype(int)


class Agent:
    def __init__(self):
        #self.env = gym.make(ENV_NAME)
        self.env = env
        self.state = self.env.reset()
        self.values = collections.defaultdict(float) # The Q-table?!
        self.best_action = 0
        self.length_opt_act = 0
        
        self.initial_Q()
    
    def act(self, eps):
        if np.random.uniform() < eps: # eps goes from 0 to 1 over iterations
            _, action = self.max_value_action()
            self.time_same_best_action(action)
        else:
            action = self.env.action_space.sample()
        
        return action
    
    def reset(self):
        self.best_action = 0
        self.length_opt_act = 0
        self.initial_Q()
        return
    
    def max_value_action(self):
        # works a bit like argmax
        # "max_value_argmax_action"
        max_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value  = self.values[(self.state, action)]
            if max_value is None or max_value < action_value:
                max_value = action_value
                best_action = action
                self.best_action = action
        return max_value, best_action
    
    def time_same_best_action(self, action):
        if action == self.best_action:
            self.length_opt_act += 1
        else:
            self.length_opt_act = 0
        return
    
    def value_update(self, s, a, r, next_s):
        # Update states here?
        best_v, _ = self.max_value_action()
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-ALPHA) + new_val * ALPHA
        self.state = next_s


    def initial_Q(self): # Initialize the Q-table.
        for s in range(nS):
            for a in range(nA):
                action = a + np.zeros((nA))
                action_other = np.arange(0.,nA) # Opponent randomizes uniformly
                actions = np.vstack([action, action_other])
                profit = profit_n(actions.transpose())
                self.values[s, a] = (sum(profit[:,0])) / ((1-GAMMA) * nA)
        return

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
