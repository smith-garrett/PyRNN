# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:10:37 2016

@author: garrettsmith
"""

# Asynchronous-update Hopfield net
# Based on description in Hertz, Krogh, & Palmer 1991 (also Trappenberg 2010)

import numpy as np
import matplotlib.pyplot as plt

# random patterns to be learned
nunits = 100
ntsteps = 350
npats = 4
pat = 2 * np.random.binomial(1, 0.5, [nunits, npats]) - 1 # start with one pattern

# setting weights via Hebbian covariance rule
W = pat @ pat.transpose() # outer prod ~> weight matix

# set random initial state
state = np.random.uniform(-0.1, 0.1, nunits)

# Dynamics
overlap = np.zeros([ntsteps, npats]) # similarity to trained pattern
H = np.zeros(ntsteps) # energy function

for t in range(0, ntsteps):
    update_idx = np.random.choice(nunits, 1) # select one unit at random to update
    state[update_idx] = np.sign(W[update_idx, :] @ pat[:, 0])
    for curr_pat in range(npats): # calc overlap with ea. stored pattern
        overlap[t, curr_pat] = state.T @ pat[:, curr_pat] / nunits
    H[t] = (-1 / nunits) * W.dot(state).dot(state)
    

# Plotting w/ sub-plots
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(overlap)
axarr[0].set_ylim([-1, 1])
axarr[0].set_title('Overlap of current state with stored patterns')
axarr[0].set_ylabel('Overlap')
axarr[1].plot(H)
axarr[1].set_title('Energy over time')
axarr[1].set_ylabel('Energy')
axarr[1].set_xlabel('Time')