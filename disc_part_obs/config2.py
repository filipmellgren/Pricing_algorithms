#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 10:55:38 2019

@author: filip
"""
import numpy as np

# Parameters
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
nA = M
nS = M**K

PARAMS = np.array([GAMMA, ALPHA, BETA, NUM_EPISODES, K, M, nS, ITER_BREAK, CONV])

# Parameters and functions related to the economic environment:
    # Given parameters
C = 1
A = 2
A0 = 1
MU = 1/2
    # Used to calculate the below. TODO: should I use a finer grid to calculate the below?
min_price_tmp = 0
price_range_tmp = 2

def profit_n(action_n, nA, c, ai, aj, a0, mu, price_range, min_price):
    '''
    profit_n gives profits in the market after taking prices as argument
    INPUT
    action_n.....an np.array([]) containing two prices
    OUTPUT
    profit.......profit, an np.array([]) containing profits
    '''
    a = np.array([ai, aj])
    a_not = np.flip(a) # to obtain the other firm's a
      
    p = (price_range * action_n/nA) + min_price # minus 1 ? was a comment I left in the other file
    p_not = np.flip(p) # to obtain the other firm's p
    num = np.exp((a - p)/mu)
    denom = np.exp((a - p)/(mu)) + np.exp((a_not - p_not)/(mu)) + np.exp(a0/mu)
    quantity_n = num / denom
          
    profit = quantity_n * (p-c)
    return(profit)


profits = np.zeros((nA, nA))

for p1 in range(nA):
    for p2 in range(nA):
        profits[p1][p2] = profit_n(np.array([p1,p2]), nA, C, A, A, A0, MU, price_range_tmp, min_price_tmp)[0]

best_response = np.zeros((nA))
for p2 in range(nA):
    best_response[p2] = np.argmax(profits[:, p2])

best_response = np.vstack((best_response, np.arange(nA))).transpose()

Nash = best_response[:,0] == best_response[:,1]

NASH_ACTION = np.argmax(Nash)
NASH_PRICE = (price_range_tmp * NASH_ACTION/(nA-1)) + min_price_tmp # minus 1?
NASH_PROFIT = profit_n(np.array((NASH_ACTION, NASH_ACTION)), nA, C, A, A, A0, MU, price_range_tmp, min_price_tmp)
MIN_PROFIT = np.min(profits)
MAX_PROFIT = np.max(profits)
MAX_ACTION = np.argmax(best_response, axis = 0)[0] # highest action ever rational to take
MAX_PRICE = (price_range_tmp * MAX_ACTION/(nA-1)) + min_price_tmp # Circular?
MONOPOLY_ACTION = np.argmax(np.diag(profits))
MONOPOLY_PRICE = (price_range_tmp * MONOPOLY_ACTION/(nA-1)) + min_price_tmp
MONOPOLY_PROFIT = np.max(np.diag(profits)) # max profit w. constraint there's 1 price

#PROFIT_NASH = 0.1133853
#PROFIT_MONOPOLY = 0.1157638717582288
MIN_PRICE = 0.9 * NASH_PRICE
MAX_PRICE = 1.1 * MONOPOLY_PRICE
price_range = MAX_PRICE- MIN_PRICE
NREWS = 15

ECON_PARAMS = np.array([C, A, A0, MU, MIN_PRICE, price_range,
                        NASH_PROFIT, MONOPOLY_PROFIT, MAX_PROFIT, MIN_PROFIT,
                        NREWS])
    
def avg_profit_gain(avg_profit):
    '''
    avg_profit_gain() gives an index of collusion
    INPUT
    avg_profit......scalar. Mean profit over episodes.
    OUTPUT
    apg.............normalised value of the scalar
    '''
    apg = (avg_profit - NASH_PROFIT) / (MONOPOLY_PROFIT - NASH_PROFIT)
    return apg

def rew_to_int(reward):
    rewrange = MAX_PROFIT - MIN_PROFIT
    rewint = np.round(NREWS * (reward-MIN_PROFIT)/rewrange).astype(int)
    return(rewint)

def to_s(act, reward):
    '''

    '''
    return(act*NREWS + reward)

