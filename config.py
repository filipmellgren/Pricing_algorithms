#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:38:07 2019

@author: filip

This file is used to set up basic parameter values that are used for this 
project.
"""
import numpy as np
from bertrand_nash import BertrandNash
env = BertrandNash()
GAMMA = 0.9 # discounting factor
ALPHA = 0.1 # step size, how much of new value gets added to old value
EPSILON = 0.3 # probability of exploration. More complicated schedule needed

PARAMS = np.array([env, GAMMA, ALPHA, EPSILON])


TEST_EPISODES = 20
NUM_EPISODES = 10
