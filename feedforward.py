# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:58:45 2016

@author: garrettsmith
"""

# Feedforward NN
# Modified from http://iamtrask.github.io/2015/07/27/python-network-part2/

import numpy as np
import matplotlib.pyplot as plt

# Inputs X, target outputs y
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]) # XOR, bias = 3. term
y = np.array([[0, 1, 1, 0]]).T

lrate = 0.1
ninput = X.shape[1]
nhid = 4
noutput = y.shape[1]
whi = 0.5 * np.random.randn(ninput, nhid)
woh = 0.5 * np.random.randn(nhid, noutput)
nepochs = 10000
errhist = np.zeros(nepochs)

for epoch in range(nepochs):
    ah = 1 / (1 + np.exp(-(np.dot(X, whi))))
    ao = 1 / (1 + np.exp(-(np.dot(ah, woh))))
    errhist[epoch] = (ao - y).T @ (ao - y)
    ao_delta = (ao - y) * (ao * (1 - ao))
    ah_delta = (ao_delta.dot(woh.T) * (ah * (1 - ah)))
    woh -= (lrate * ah.T.dot(ao_delta))
    whi -= (lrate * X.T.dot(ah_delta))

print(ao)
plt.plot(errhist)
plt.ylabel('Sum squared error')
plt.xlabel('Training epoch')
