#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:53:20 2019

@author: filip
"""

import gym
from gym import spaces
import numpy as np

class PrisonerDilemma(gym.Env): 
    # useful blog post:
    # https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
  """
  similar to BertrandNash but simplified to represent a prisoner dilemma 
  scenario
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
    # TODO: decide what action space to use and what type of competition to model.
    self.action_space = spaces.Discrete(2) # Page 400 DRL hands on. POSSIBILITY IT IS DISCRETE, two types of prices
    self.observation_space = spaces.Discrete(4) # Maybe, own action in the last period should be observed as well?
        # Add to obs space, own action last period, periods of punishment
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
          self.quantity_n =  np.array([-1,2])
      elif all(p == np.array([0,1])):
          self.quantity_n =  np.array([2,-1])
      elif all(p == np.array([1,1])):
          self.quantity = np.array([1,1])
      else:
          print(p)
      return