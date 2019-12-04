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
      Can self.attributes be thought of as endogenous and action as exogenous?
      
      Demand is stochastic:
          prob alpha, zero demand
          prob (1-alpha), high demand state

      https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
  """
  metadata = {'render.modes': ['human']} #?
# Set this in SOME subclasses
#  reward_range = (-float('inf'), float('inf'))
 # spec = None
  # Set these in ALL subclasses
 # action_space = None # TODO set when possible to test
  #observation_space = None # TODO set when possible to test

  def __init__(self):
    # Define action and observation space
    # They must be gym.spaces objects
    
    # inherit class methods from another gym environment
    # Initialize object with values here
    # Actions are setting a price inside the interval
    #self.action_space = spaces.Box(
    #        low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float16) # can be of form: np.array([0, 0]). TODO: High should be infinity!
    # An instance of something else (Box environment)

    # Action space can be bounded between Bertrand Nash equilibrium and monopoly price
    self.action_space = spaces.Box(low=0, high=9, shape=(2, 1), dtype=np.int_)
    self.observation_space = spaces.Box(
            low=0, high=24, shape=(2, 1), dtype=np.int)
    
   #? self.state_size = 1
   #? self.profit = 0

    
    super().__init__()
#    super().all(*args, **kwargs)
    
    # Markov game: 
    #https://stackoverflow.com/questions/44369938/openai-gym-environment-for-multi-agent-games
  '''
    def step(self, action, player):
      # Execute one time step within the environment
      self._take_action(action, player)
      self.current_step += 1
      reward = self.profit  # has to be specific to a player
      done = self.current_step > 100
      obs = self._next_observation(action)
      return obs, reward, done, {}
      
      
      action_space: The Space object corresponding to valid actions
      observation_space: The Space object corresponding to valid observations
      reward_range: A tuple corresponding to the min and max possible rewards
  '''

  def step(self, action_n):
      # action_n..... list of actions taken by the agents
      
      # set action for each agent
      '''
      for i, agent in enumerate(self.agents):
          self._take_action(action_n[i], agent)
      ''' 
      self._take_action(action_n)
      obs_n    = self._next_observation()
      reward_n = self.profit_n
      done =  self.current_step > 100 # TODO: don't hardcode
      #info_n   = {'n': []} # TODO: what is this?
      self.current_step += 1
      return obs_n, reward_n, done, {}
          
  def reset(self):
      # Reset the state of the environment to an initial state
      self.acc_profit_n = list((0,0)) # TODO: don't hardcode
      self.current_step = 0
      return self.acc_profit_n
  
  def _next_observation(self):
      '''
      Makes it explicit that what's observed is the profit
      INPUT:
          self.profits
      '''
      obs = self.profit_n
      return obs

  def _take_action(self, action_n):
      '''
      _take_action() gives values of the market given the action
      The calculation of profits assumes mc = 0 <==> Pi = pq
      INPUT:
          action_n..........list of actions (prices) set by the agents.
      OUTPUT:
          self.quantity_n...list of the quantities sold in the market.
          self.profit_n......list of profits earned in the market.
          self.acc_profit_n..list of the agents total accumulated profits (max. objective)
      '''
      self.quantity_n = self.demand(action_n)
      self.profit_n = np.asarray(self.quantity_n) * np.asarray(action_n) # TODO: break out into its own method, like quantities
      self.acc_profit_n += self.profit_n # Objective function
        
  def render(self, mode='human', close=False):
      # Render the environment to the screen
      print(f'Step: {self.current_step}')
      print(f'Profit: {self.profit}')
      print(f'Acc Profit: {self.acc_profit}')
      
  def demand(self, action_n):
      '''
      demand() calculates demand for each agent's product given actions prices)
        works for 2 firms. For n firms, denom is a longer sum for j = 1...n
      INPUT:
          action_n.......list of actions (prices) taken by the agents
          quality........list with index of product qualities
          mu.............index of horizontal differentiation
      OUTPUT: 
          quantities.....list of quantities, one for each agent. 
      '''
      quantities = list(10 - np.asarray(action_n)) # TODO: add complexity once system is working
      #pi = action_n[0]
      #pj = action_n[1]
      #num = np.exp((ai - pi)/mu)
      #denom = np.exp((ai - pi)/(mu)) + np.exp((aj - pj)/(mu)) + np.exp(a0/mu)
      #quantities = num / denom
      return quantities
      
      