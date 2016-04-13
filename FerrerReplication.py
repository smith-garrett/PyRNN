# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:21:45 2016

@author: garrettsmith
"""

# Replication of Ferrer i Concho & Sole (PNAS, 2003)
# n signals, m referents
# 

import numpy as np

	
class FormMeaningMat:
	def __init__(self, nsigns, nrefs):
		self.nsigns = nsigns
		self.nrefs = nrefs
		self.refMat = np.zeros(nrefs, nsigns)
		
	def initMat(self):
		self.refMat = np.random.binomial(1, 0.5, [self.nrefs, self.nsigns])
		
	def learn(self):
		
		
if __name__ == '__main__':
	print('Hello')