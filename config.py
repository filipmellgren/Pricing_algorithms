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
ALPHA = 0.02 # step size, how much of new value gets added to old value
BETA = 4*0.00001 # Parameter for epsilon greedy approach. Lower leads to more exploration
NUM_EPISODES = 1000 # same as n.o. sessions in Calvano
K = 1 # Length of memory. I.e. remembering the last step
M = 15 # n.o. actions.  Makes the discrete environment the prisoner dilemma (they mostly use 15)

PARAMS = np.array([GAMMA, ALPHA, BETA, NUM_EPISODES, K, M])

c = 1
a = 2
a0 = 1
mu = 1/2
min_price = 1.6
max_price = 1.74
price_range = max_price- min_price
ECON_PARAMS = np.array([c, a, a0, mu, min_price, price_range])

