# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:57:11 2016

@author: garrettsmith
"""

# Online exercise from Dayan & Abbott 2001
# Exercise 4 from Ch. 7

# excitatory firing rate re = -re + mee * re + mei * ri - ge
# inhibitory firing rate ri = -ri + mii * ri + mie * re - gi

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# declaring constants:
mee = 1.25
mie = 1.0
mii = 0.0
mei = -1.0
ge = -10 # Hz
gi = 10 # Hz
te = 10 # ms
#ti = np.array([20, 50])
ti = np.arange(36, 39, 1)

tau0 = 0.01
nsec = 20
tvec = np.arange(0, nsec, tau0)

# Euler forward integration
#re = np.zeros(nsec * tau0**-1)
#re[0] = 0.1
#ri = np.zeros(nsec * tau0**-1)
#ri[0] = 0.1

for timectl in range(0, len(ti)):
	re = np.zeros(nsec * tau0**-1)
	re[0] = 0.1
	ri = np.zeros(nsec * tau0**-1)
	ri[0] = 0.1
	
	for tstep in range(1, len(tvec)):
		re[tstep] = re[tstep - 1] + (te**-1) * (-re[tstep - 1] + mee * re[tstep - 1] + mei * ri[tstep - 1] - ge)
		ri[tstep] = ri[tstep - 1] + (ti[timectl]**-1) * (-ri[tstep - 1] + mii * ri[tstep - 1] + mie * re[tstep - 1] - gi)
	
#	plt.plot(tvec, re, color = 'blue', label = 'rE')
#	plt.plot(tvec, ri, color = 'green', label = 'rI')
#	plt.legend(loc = 2)
#	plt.title('tI = {}'.format(ti[timectl]))
	plt.plot(re, ri)
	plt.ylabel('rI')
	plt.xlabel('rE')
	plt.show()
