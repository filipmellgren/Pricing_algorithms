#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 22 2019

@author: Filip Mellgren

"""

# Discrete case
import gym
from gym import spaces
import numpy as np
from gym.envs.toy_text import discrete
from config2 import ECON_PARAMS
from config2 import PARAMS
from config2 import profit_n

C = ECON_PARAMS[0]
AI = ECON_PARAMS[1]
AJ = ECON_PARAMS[1]
A0 = ECON_PARAMS[2]
MU = ECON_PARAMS[3]
MIN_PRICE = ECON_PARAMS[4]
PRICE_RANGE = ECON_PARAMS[5]
MAX_REWARD = ECON_PARAMS[8]
MIN_REWARD = ECON_PARAMS[9]
NREWS = ECON_PARAMS[10].astype(int)

nA = PARAMS[5].astype(int) # Number of unique prices

class DiscrBertrand(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']} #?
    
    # useful blog post:
    # https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
    """
    This environment represents a discrete world with two agents. The agents 
    may set prices which generates profits depending on the joint behaviour.
      
    In principle, the environment is similar to FrozenLake with the difference 
    that rows and columns (the state) are prices in the previous period.
      
    Inherits from discrete.Discrete which:
          
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
    P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self):
        # self.nrow, self.ncol = nrow, ncol = desc.shape
        nrow = nA # Number of own possible actions, last state
        ncol = nA
        nS = nrow * nA
        isd = np.zeros(nS)
        P = {s : {a : [] for a in range(nA*nA)} for s in range(nS)} # Takes both actions into account

        def to_s(row, col):
            '''
            enumerates row, col combo to a state. the row and col can be thought
            of as action and profit in the last period in this application.
            '''
            return(row*ncol + col)

        for s in range(nS): # TODO: Correct?
            for action1 in range(nA): # TODO: can this be simplified?
                for action2 in range(nA):
                    a = to_s(action1, action2)
                    action_n = np.array([action1, action2])
                    li = P[s][a]
                    reward_n = profit_n(action_n, nA, C, AI, AJ, A0, MU, PRICE_RANGE, MIN_PRICE)
                    newstate = to_s(action1, action2) # new env state is determined by what they did in the last period. TODO: is it even important whatexaclty it is?
                    done = False # No need to update done at init – my stopping rule does not depend on state
                    # Here, P[s][a] is not updated
                    li.append((1.0, newstate, reward_n, done)) # Why does it not need "P[s][a].append"?
                    # Here, P[s][a] is updated
        super(DiscrBertrand, self).__init__(nS, nA, P, isd)


