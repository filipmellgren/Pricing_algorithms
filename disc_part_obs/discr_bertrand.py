#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:53:20 2019

@author: filip

# TODO: build on top of dicrete.Discrete
"""

# Discrete case
import gym
from gym import spaces
import numpy as np
from gym.envs.toy_text import discrete
from config import ECON_PARAMS
from config import PARAMS

C = ECON_PARAMS[0]
AI = ECON_PARAMS[1]
AJ = ECON_PARAMS[1]
A0 = ECON_PARAMS[2]
MU = ECON_PARAMS[3]
MIN_PRICE = ECON_PARAMS[4]
PRICE_RANGE = ECON_PARAMS[5]

nA = PARAMS[5].astype(int) # Number of unique prices

class PrisonerDilemma(discrete.DiscreteEnv):  # maybe have to drop gym.Env
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
  
  
  metadata = {'render.modes': ['human']} #?

  def __init__(self):

      # self.nrow, self.ncol = nrow, ncol = desc.shape
      nrow = nA # Number of own possible actions, last state
      ncol = nA # Number of other's possible actions, last state
      nS = nrow * ncol
      isd = np.zeros(nS)
      P = {s : {a : [] for a in range(nA*nA)} for s in range(nS)} # Takes both actions into account
      
      def to_s(row, col):
          '''
          enumerates row, col combo to a state. the row and col can be thought
          of as actions in the last period in this application
          '''
          return(row*ncol + col)
          
      def profit_n(action_n):
          a = np.array([AI, AJ])
          a_not = np.flip(a) # to obtain the other firm's a
          
          p = (PRICE_RANGE * action_n/(nA-1)) + MIN_PRICE
          
          p_not = np.flip(p) # to obtain the other firm's p
          num = np.exp((a - p)/MU)
          denom = np.exp((a - p)/(MU)) + np.exp((a_not - p_not)/(MU)) + np.exp(A0/MU)
          quantity_n = num / denom
          
          profit = quantity_n * (p-C)
          return(profit)
          
          # Necessary to loop over everything in here?
          # TODO: what does this even do?
          # Update: P is indeed updated in the loop, albeit I do not see how
          # But what is it used for?
          # Think: it gives the transitions for a given action in the step method
      for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for action1 in range(nA): 
                    for action2 in range(nA):
                        a = to_s(action1, action2)
                        action_n = np.array([action1, action2])
                        li = P[s][a]
                        newstate = to_s(action1, action2)
                        done = False # No need to update done at init – my stopping rule does not depend on state
                        reward_n = profit_n(action_n)
                        # Here, P[s][a] is not updated
                        li.append((1.0, newstate, reward_n, done)) # Why does it not need "P[s][a].append"?
                        # Here, P[s][a] is updated
      
      super(PrisonerDilemma, self).__init__(nS, nA, P, isd)
    