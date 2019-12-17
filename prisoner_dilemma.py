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

class PrisonerDilemma(discrete.DiscreteEnv):  # maybe have to drop gym.Env
    # useful blog post:
    # https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
  """
  Similar to BertrandNash but simplified to represent a discrete scenario
  
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
      nrow = 20
      ncol = 1
      nA = 4
      nS = nrow * ncol
      isd = np.zeros(nS)
      P = {s : {a : [] for a in range(nA)} for s in range(nS)}
      super(PrisonerDilemma, self).__init__(nS, nA, P, isd)
        #    super().all(*args, **kwargs)
    
    # Markov game: 
    #https://stackoverflow.com/questions/44369938/openai-gym-environment-for-multi-agent-games
x = np.random.randint(0,2, 10)
y = np.random.randint(0,2, 10)
np.array(x == y).astype('float64').ravel()

  def step(self, action_n):
      # agent gives action
      # info about outcome is returned (next obs, reward, end flag)
      # action_n..... list of actions taken by the agents
      
      # set action for each agent
      '''
      for i, agent in enumerate(self.agents):
          self._take_action(action_n[i], agent)
      ''' 
      
      self.demand(action_n) # sets self.quantity_n
      self.profit_n = np.asarray(self.quantity_n) * np.asarray(action_n) #maybe: break out into its own method, like quantities
      reward_n = self.profit_n
      self.state_n = reward_n
      done =  self.current_step > 100 # TODO: don't hardcode
      #info_n   = {'n': []} # TODO: what is this?
      self.current_step += 1
      return self.state_n, reward_n, done, {}
  # numpy matrix, float value of reward, boolean, extra info
          
  def reset(self):
      # Reset the state of the environment to an initial state
      self.current_step = 0
      self.profit_n = np.asarray([0,0]) # TODO: Set up smarter
      self.state_n = self.profit_n 
      
        
  def render(self, mode='human', close=False):
      # Render the environment to the screen
      print(f'Step: {self.current_step}')
      print(f'Profit: {self.profit}')
      

  def interact(self, action_n):
      old_state_n = self.state_n
      new_state_n, reward_n, is_done, _ = self.step(action_n)
      self.state_n = self.reset() if is_done else new_state_n
      return(old_state_n, action_n, reward_n, new_state_n)
      
  def demand(self, action_n):
      '''
      demand() calculates demand for each agent's product given actions prices)
        WORKS FOR 2 FIRMS. For n firms, denom is a longer sum for j = 1...n
      INPUT:
          action_n.......array of actions (prices) taken by the agents
          quality........list with index of product qualities
          mu.............index of horizontal differentiation
      OUTPUT: 
          quantities.....list of quantities, one for each agent. 
      '''
# TODO: add complexity once system is working
      # OR, make it flexible within the Discrete environment
      # Parameters used, might want to keep them at the top of the class, outside the function or pass as arguments
      '''
      a0 = 1 
      ai = 2
      aj = 2
      mu = 1
      a = np.array([ai, aj])
      a_not = np.flip(a) # to obtain the other firm's a
      p = action_n
      p_not = np.flip(p) # to obtain the other firm's p

      action_n
      num = np.exp((a - p)/mu)
      denom = np.exp((a - p)/(mu)) + np.exp((a_not - p_not)/(mu)) + np.exp(a0/mu)
      self.quantity_n = num / denom
      '''
      p = action_n
      if all(p == np.array([0,0])):
          self.quantity_n =  np.array([0,0])
      elif all(p == np.array([1,0])):
          self.quantity_n =  np.array([0,2])
      elif all(p == np.array([0,1])):
          self.quantity_n =  np.array([2,0])
      elif all(p == np.array([1,1])):
          self.quantity = np.array([50,50])
      else:
          print(p)
      return