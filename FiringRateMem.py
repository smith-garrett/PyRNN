# -*- coding: utf-8 -*-

"""
# Goal: short term memory in network of firing rate neurons
# Dynamics:
# tau_r * dr_i(t) / dt = -r_i + F(input + sum(w_ij * r_j) + bias)

# Still not working quite as desired. Look at lit. on continuous attr. nets
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#def dyn(r, t, act_fn, W):
def dyn(r_t, t, W):
    #drdt = -r_t + rect_lin(np.dot(W, r_t.transpose()))
    drdt = -1 * r_t + sig(np.dot(W, r_t.transpose()))
    return drdt
    
def rect_lin(vec): # rectified linear activation fn.
    vec[vec < 0] = 0
    return vec
    
def sig(vec):
    return 1 / (1 + np.exp(-vec))

if __name__ == '__main__':
    nnodes = 10
    r0 = np.zeros(nnodes)
    r0[5] = 1.0
    W = np.zeros((nnodes, nnodes))
    vec = np.arange(0, nnodes, 1)
    for row in range(0, W.shape[0]):
        W[row, :] = -(vec - row)**2 + 1.2
        W[row, row] = 0.0
            
    #W = np.zeros((nnodes, nnodes))
    #for n_from in range(0, nnodes):
    #    for n_to in range(0, nnodes):
    #        W[n_from, n_to] = -J0 + np.cos((2 * np.pi * (n_to - n_from)) / (nnodes))
    
    # Run net:
    tvec = np.arange(0, 20, 0.1)
    r_t = odeint(dyn, r0, tvec, (W,))
    
    # Plot:
    f, axarr = plt.subplots(nnodes, sharex = True)
    for ax in range(0, nnodes):
        axarr[ax].plot(tvec, r_t[:, ax])
#        axarr[ax].set_title('a = {}, b = {}'.format(a, b))
        axarr[ax].set_ylabel('n{}'.format(ax))
    axarr[nnodes - 1].set_xlabel('Time')
    plt.show()

#    plt.plot(r_t)
 #   plt.show()