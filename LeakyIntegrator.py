# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:44:20 2015

@author: garrettsmith
"""

## Goal: simple, leaky integrator neuron

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sig(act):
    return 1 / (1 + np.exp(-act))
        
def dyn(act, t, tau, inputs, W):
    dact_dt = -tau * act + sig(np.dot(W, inputs.transpose()))
    return dact_dt
        
if __name__ == '__main__':
    tvec = np.arange(0, 20, 0.1)
    act0 = 1.0
    W = np.array([0.1, 5.])
    inputs = np.array([2., 5.])
    act = odeint(dyn, act0, tvec, (0.1, inputs, W,))
    plt.plot(tvec, act)
    plt.show()