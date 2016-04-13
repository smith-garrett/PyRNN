# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:58:45 2016

@author: garrettsmith
"""

# Feedforward NN

import numpy as np

def sig(x):
	return 1. / (1. + np.exp(-x))
	
def sigprime(x):
	return x * (1. - x)