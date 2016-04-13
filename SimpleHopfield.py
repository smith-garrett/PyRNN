# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:42:15 2016

@author: garrettsmith
"""

# Hopfield/Amit net from Trappenberg 2010

import numpy as np
import matplotlib.pyplot as plt

# random patterns to be learned
nunits = 2005
ntsteps = 10
npats = 10
pat = 2 * np.random.binomial(1, 0.5, [nunits, npats]) - 1

# setting weights via Hebbian covariance rule
w = pat @ pat.transpose() # outer prod ~> weight matix

# set random initial state
state = np.zeros([nunits, ntsteps])
state[:, 0] = np.random.uniform(-0.1, 0.1, nunits)

# run the net:
for t in np.arange(1, ntsteps):
	state[:, t] = np.sign(w @ state[:, t - 1])


# calc energy over time:
H = np.zeros(ntsteps)
for t in range(0, ntsteps):
	H[t] = (-1 / nunits) * w.dot(state[:, t]).dot(state[:, t])

# set up sub-plots
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot((state.transpose() @ pat) / nunits)
axarr[0].set_ylim([-1, 1])
axarr[0].set_title('Overlap of current state with stored patterns')
axarr[0].set_ylabel('Overlap')
axarr[1].plot(H)
axarr[1].set_title('Energy over time')
axarr[1].set_ylabel('Energy')
axarr[1].set_xlabel('Time')

# plot overlap of current state w/ each of the trained patterns
#plt.plot((state.transpose() @ pat) / nunits)
#plt.axis([0, ntsteps, -1, 1])
#plt.show()

# Plot energy
#plt.plot(H)
