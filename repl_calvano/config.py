#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:38:07 2019

@author: filip

This file is used to set up basic parameter values that are used for this 
project.
"""
import numpy as np
# =============================================================================
# from bertrand_nash import BertrandNash
# from prisoner_dilemma import PrisonerDilemma
# env = PrisonerDilemma()
# # Parameters set so as to match Calvano et al. See page 19.
# =============================================================================
GAMMA = 0.95 # discounting factor
ALPHA = 0.1 # learning rate
BETA = 0.5*10**(-5) # Parameter for epsilon greedy approach. Lower leads to more exploration
#NUM_EPISODES = 1000 # same as n.o. sessions in Calvano
NUM_EPISODES = 5 # testing purposes
ITER_BREAK = 2*10**6 # Calvano uses 10**9
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
# TODO: derive these better:
MIN_PRICE = 1.6
MAX_PRICE = 1.74
price_range = MAX_PRICE- MIN_PRICE
# TODO: derive analytically and more generally
PROFIT_NASH = 0.113203 
PROFIT_MONOPOLY = 0.1157638
ECON_PARAMS = np.array([C, A, A0, MU, MIN_PRICE, price_range,
                        PROFIT_NASH, PROFIT_MONOPOLY])

AI = A
AJ = A

def profit_n(action_n):
    '''
    profit_n gives profits in the market after taking prices as argument
    INPUT
    action_n.....an np.array([]) containing two prices
    OUTPUT
    profit.......profit, an np.array([]) containing profits
    '''
    a = np.array([AI, AJ])
    a_not = np.flip(a) # to obtain the other firm's a
      
    p = (price_range * action_n/M) + MIN_PRICE 
    p_not = np.flip(p) # to obtain the other firm's p
    num = np.exp((a - p)/MU)
    denom = np.exp((a - p)/(MU)) + np.exp((a_not - p_not)/(MU)) + np.exp(A0/MU)
    quantity_n = num / denom
          
    profit = quantity_n * (p-C)
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
