"""
Created on Mon Sep 21 08:35:18 2015

@author: garrettsmith
"""

# basic feedforward net

import numpy as np
from pylab import *

class FeedForwardNet:
    def __init__(self, nin, nhid, nout):
        self.lrate = 0.01
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        
        # Initialize weights in N(0, 0.1)
        self.w_hi = 0.1 * np.random.randn(self.nhid + 1, self.nin + 1) # plus bias
        self.w_oh = 0.1 * np.random.randn(self.nout + 1, self.nhid + 1)
        
        # Init activations
        self.act_i = np.zeros((self.nin + 1, 1), dtype = float)
        self.act_h = np.zeros((self.nhid + 1, 1), dtype = float)
        self.act_o = np.zeros((self.nout + 1, 1), dtype = float)
        
        # Init deltas
        self.delta_h = np.zeros((self.nhid, 1), dtype = float)
        self.delta_o = np.zeros((self.nout, 1), dtype = float)
        
    def forward(self, input):
        # Set input
        self.act_i[:-1, 0] = input
        self.act_i[-1, 0] = 1. # bias = 1
        
        # Input -> hidden
        self.act_h = np.tanh(np.dot(self.w_hi, self.act_i))
        self.act_h[-1, 0] = 1 # set bias
        
        # Hidden -> out
        self.act_o = np.tanh(np.dot(self.w_oh, self.act_h))
        
    def backward(self, targets): # need to revise this...
        error = self.act_o - np.array(targets, dtype=float) # not correct error
         
        # deltas of output neurons
        self.delta_o = (1 - np.tanh(self.act_o)**2) * error
                 
        # deltas of hidden neurons
        self.delta_h = (1 - np.tanh(self.act_h)**2) * np.dot(self.w_oh.transpose(), self.delta_o)
                 
        # apply weight changes
        self.w_hi = self.w_hi - self.lrate * np.dot(self.delta_h, self.act_i.transpose()) 
        self.w_oh = self.w_oh - self.lrate * np.dot(self.delta_o, self.act_h.transpose())
        return error
    
    def getOutput(self):
        return self.act_o

if __name__ == '__main__':
    ''' 
    XOR test example for usage of ffn
    '''
     
    # define training set
    xorSet = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xorTeach = [[0], [1], [1], [0]]
     
    # create network
    ffn = FeedForwardNet(2, 2, 1)
     
    count = 0
    error = np.zeros([20000, 1])
    while(count <= 20000):
        # choose one training sample at random
        rnd = np.random.randint(0,4)
         
        # forward and backward pass
        ffn.forward(xorSet[rnd])
#        error[count,:] = np.sum(ffn.backward(xorTeach[rnd]))
        ffn.backward(xorTeach[rnd])
         
        # output for verification
        print(count, xorSet[rnd], ffn.getOutput())#, 
        if ffn.getOutput()[0] > 0.9:
            print('TRUE')
        elif ffn.getOutput()[0] < 0.1:
            print('FALSE')
        count += 1
        
#    plot(error)