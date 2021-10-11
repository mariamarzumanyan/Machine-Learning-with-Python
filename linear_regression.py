# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:47:21 2020

@author: Mariam Arzumanyan
Title:  Linear Regression
"""

import numpy as np
from pylab import *
import matplotlib.pyplot as plt
#%matplotlib inline 

print('You\'re running python %s' % sys.version.split(' ')[0])

N = 40 # 
X = np.random.rand(N,1) # Sample N points randomly along X-axis
X=np.hstack((X,np.ones((N,1))))  # Add a constant dimension
w = np.array([3, 4]) # defining a linear function 
y = X@w + np.random.randn(N) * 0.1 # defining labels

plt.plot(X[:, 0],y,".")

w_closed = np.linalg.inv(X.T@X)@X.T@y
plt.plot(X[:, 0],y,".") # plot the points
z=np.array([[0,1],      # define two points with X-value 0 and 1 (and constant dimension)
            [1,1]])
plt.plot(z[:,0], z@w_closed, 'r') # draw line w_closed through these two points


#The solution is the same, but it is typically faster and more stable in case  (𝐗𝑇𝐗)  is # not 

w_closed = np.linalg.solve(X.T@X,X.T@y) 

