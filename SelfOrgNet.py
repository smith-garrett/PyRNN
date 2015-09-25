# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 08:31:10 2015

@author: garrettsmith
"""

# Idea: create extensibe, reusable self-organizing neural net system a la
# Tabor et al. 2013, Kukona et al. 2013

# Structure of net: set of input units feedforward-connected to recurrently
# connected, continuously settling localist output units

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SelfOrgNet:
    def __init__(self, n_inputs, n_outputs, tau_a = 0.1, tau_l = 0.01, max_time = 10.):
        self.tau_a = tau_a # time constant for activation change
        self.tau_l = tau_l # time constant for weight change
        self.max_time = max_time
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        # Create weights
        #self.w_oi = np.zeros([self.n_inputs, self.n_outputs])
        #self.w_oo = np.zeros([self.n_outputs, self.n_outputs])
        self.w_oi = np.random.uniform(-0.1, 0.1, [self.n_inputs, self.n_outputs])
        self.w_oo = np.random.uniform(-0.1, 0.1, [self.n_outputs, self.n_outputs])
        
        # Initialize activations
        self.act_i = np.ones(self.n_inputs)
        self.net = np.zeros(self.n_outputs)
        #self.act_o = np.zeros(self.n_outputs)
        self.act_o = np.random.uniform(-0.5, 0.5, self.n_outputs)
        self.act_o_hist = np.zeros(self.n_outputs)
        
    def set_inputs(self, inputs):
        self.act_i = inputs
        
    def act_fn(self, act_o, t):
        self.net = self.w_oi.dot(self.act_i.T) + self.w_oo.dot(self.act_o.T)
        return self.tau_a * self.net * act_o * (1 - act_o)

    def run_net(self):
        tvec = np.arange(0.0, self.max_time, 0.01)
        self.act_o_hist = odeint(self.act_fn, self.act_o, tvec)
        return self.act_o_hist
        
    def plot_activations(self):
        tvec = np.arange(0.0, self.max_time, 0.01)
        f, axarr = plt.subplots(self.n_outputs, sharex = True)
        for output in range(0, self.n_outputs):
            axarr[output].plot(tvec, self.act_o_hist[:,output])
            axarr[output].set_ylabel('Output node {}'.format(output + 1))
            
        axarr[0].set_title('Output activations')
        axarr[self.n_outputs - 1].set_xlabel('Time')
        plt.show()


if __name__ == '__main__':
    ''' 
    Self-organizing, continuous time neural network
    '''
    
    # Logical NOT on a 2d array
    targ_array = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    
    so1 = SelfOrgNet(n_inputs = 2, n_outputs = 2, tau_a = 0.1, max_time = 50.)
    so1.set_inputs(np.array([3., 2.]))
    so1.run_net()
    so1.plot_activations()