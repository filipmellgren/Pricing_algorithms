#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:07:10 2019

@author: filip
"""

import gym
from gym import spaces
import numpy as np
import random

class BertrandNash(gym.Env): # give it the Box environment
  """Custom Environment that follows gym interface
      This class describes a market characterised by Bertrand behaviour
      
      Demand is stochastic:
          prob alpha, zero demand
          prob (1-alpha), high demand state

      https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
  """
  metadata = {'render.modes': ['human']} #?

  def __init__(self):
    # Define action and observation space
    # They must be gym.spaces objects
    
    # inherit class methods from another gym environment
    # Initialize object with values here
    # Actions are setting a price inside the interval
    #self.action_space = spaces.Box(
    #        low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float16) # can be of form: np.array([0, 0]). TODO: High should be infinity!
    # An instance of something else (Box environment)
  #  self.observation_space = spaces.Box() # What can we observe? Demand function?
        # Actions of the format Buy x%, Sell x%, Hold, etc.
    self.action_space = spaces.Box(low=0, high=9, shape=(1, 1), dtype=np.int_)
    
    # Prices contains the OHCL values for the last five prices
    self.observation_space = spaces.Box(
      low=0, high=1, shape=(6, 6), dtype=np.float16)
    
    self.state_size = 1
    self.profit = 0

    
    super().__init__()
#    super().all(*args, **kwargs)
    
  def step(self, action):
      # Execute one time step within the environment
      self._take_action(action)
      self.current_step += 1
      reward = self.profit 
      done = self.current_step > 100
      obs = self._next_observation(action)
      return obs, reward, done, {}
    
  def reset(self):
      # Reset the state of the environment to an initial state
      self.acc_profit = 0
      self.current_step = 0 # ??
      return 0
  
  def _next_observation(self, action):
      #obs = np.array([self.profit])
      obs = action
      return obs # TODO: next_observation needs to be an int

  def _take_action(self, action):
      # Here is where the agent decides how much to charge
      self.demand = 10 - action
      self.profit = self.demand * action
      self.acc_profit += self.profit
        
  def render(self, mode='human', close=False):
      # Render the environment to the screen
      print(f'Step: {self.current_step}')
      print(f'Profit: {self.profit}')
      print(f'Acc Profit: {self.acc_profit}')