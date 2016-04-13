# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:13:19 2015

@author: garrettsmith
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz(u, t, a, b, c):
    u[0] = a * (u[1] - u[0])
    u[1] = u[0] * (c - u[2]) - u[1]
    u[2] = u[0] * u[1] - b * u[2]
    return u
    
if __name__ == '__main__':
    a = 10.
    b = 28.
    c = 8/3.
    u0 = np.array([2.0, 3.0, 4.0])
    tspan = np.arange(0, 10, 0.01)
    udot = odeint(lorenz, u0, tspan, (a, b, c,))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot(udot[:,0], udot[:,1], udot[:,2])
    plt.show()