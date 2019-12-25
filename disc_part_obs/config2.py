#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 10:55:38 2019

@author: filip
"""

import numpy as np
GAMMA = 0.95 # discounting factor
ALPHA = 0.1 # learning rate
BETA = 0.5*10**(-5) # Parameter for epsilon greedy approach. Lower leads to more exploration
#NUM_EPISODES = 1000 # same as n.o. sessions in Calvano
NUM_EPISODES = 5 # testing purposes
ITER_BREAK = 10**5 # Calvano uses 10**9
# Calvabo on number of iterations:
# =============================================================================
# But convergence requires a very large number
# of periods, on the order of hundreds of thousands
# =============================================================================

CONV = 25000 # Calvano uses 25000
K = 1 # Length of memory. I.e. remembering the last step
M = 15 # n.o. actions.  Makes the discrete environment the prisoner dilemma (they mostly use 15)
nS = M**K

PARAMS = np.array([GAMMA, ALPHA, BETA, NUM_EPISODES, K, M, nS, ITER_BREAK, CONV])

C = 1
A = 2
A0 = 1
MU = 1/2
# TODO: derive these 
MIN_PRICE = 1.6136136136136137
MAX_PRICE = 1.7371274174174174
price_range = MAX_PRICE- MIN_PRICE
# TODO: derive analytically and more generally
PROFIT_NASH = 0.1133853
PROFIT_MONOPOLY = 0.1157638717582288
# TODO: derive more carefully
MAX_REWARD = 0.12593942529901705
MIN_REWARD = -0.8668133321973348
NREWS = 15

ECON_PARAMS = np.array([C, A, A0, MU, MIN_PRICE, price_range,
                        PROFIT_NASH, PROFIT_MONOPOLY, MAX_REWARD, MIN_REWARD,
                        NREWS])

AI = A
AJ = A

def profit_n(action_n, c, ai, aj, a0, mu, price_range, min_price):
    '''
    profit_n gives profits in the market after taking prices as argument
    INPUT
    action_n.....an np.array([]) containing two prices
    OUTPUT
    profit.......profit, an np.array([]) containing profits
    '''
    a = np.array([ai, aj])
    a_not = np.flip(a) # to obtain the other firm's a
      
    p = (price_range * action_n/M) + min_price 
    p_not = np.flip(p) # to obtain the other firm's p
    num = np.exp((a - p)/mu)
    denom = np.exp((a - p)/(mu)) + np.exp((a_not - p_not)/(mu)) + np.exp(a0/mu)
    quantity_n = num / denom
          
    profit = quantity_n * (p-c)
    return(profit)
    
def avg_profit_gain(avg_profit):
    '''
    avg_profit_gain() gives an index of collusion
    INPUT
    avg_profit......scalar. Mean profit over episodes.
    OUTPUT
    apg.............normalised value of the scalar
    '''
    apg = (avg_profit - PROFIT_NASH) / (PROFIT_MONOPOLY - PROFIT_NASH)
    return apg

def rew_to_int(reward):
    rewrange = MAX_REWARD - MIN_REWARD
    rewint = np.round(NREWS * (reward-MIN_REWARD)/rewrange).astype(int)
    return(rewint)

def to_s(act, reward): # TODO: hardcoded 15, 
    '''

    '''
    return(act*NREWS + reward)
