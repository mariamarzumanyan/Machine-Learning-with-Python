# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:07:55 2020

@author: Mariam Arzumanyan

"""

import numpy as np
import matplotlib.pyplot as plt
import sys

#Choose appropriate path
sys.path.append('XX')
from helper import *

print('You\'re running python %s' % sys.version.split(' ')[0])

import pandas as pd
def load_data(file='data.csv', label=True):
    '''
    Input:
        file: filename of the dataset
        label: a boolean to decide whether to return the labels or not
    Returns:
        X: patient attributes
        y: label (only if label=True)
        
    '''
    df=pd.read_csv(file)
    df=df.to_numpy()
    X1=df[:, :-1]
    y=df[:, -1]
    X2=df[:,:]
    if label==True:
        return X1, y
    else:
        return X2
   
df=np.genfromtxt('data.csv')
#df.columns
y=[df[ :-1]]
Xtest = load_data(file='data.csv', label=False)
Xtest.shape
Xtrain, ytrain = load_data()

assert len(Xtrain) == len(ytrain)

assert type(Xtrain)==np.ndarray
#type(Xtrain)
ntr,dtr=Xtrain.shape
nte,dte=Xtest.shape
dte

X, y = load_data()

# Create a regression with no restriction on its depth
# if you want to create a tree of depth k
# then call RegressionTree(depth=k)
tree = RegressionTree(depth=np.inf)

# To fit/train the regression tree
tree.fit(X, y)

# To use the trained regression tree to make predictions
pred = tree.predict(X)

def square_loss(pred, truth):
    return np.mean((pred - truth)**2)

def test():
    '''
        prediction: the prediction of your classifier on the heart_disease_test.csv
    '''
    prediction = None
    Xtrain, ytrain = load_data(file='heart_disease_train.csv', label=True)
    ytrain=ytrain>0
    Xtest = load_data(file='heart_disease_test.csv', label=False)
    tree = RegressionTree(depth=4)
    tree.fit(Xtrain, ytrain)
    prediction=tree.predict(Xtest)
   
    return prediction


# The following test wil check that your test function returns a loss less than 2 on a sample dataset
# ground truth:
gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

pred = test()
test_loss = square_loss(pred, gt)
print('Your test loss: {:0.4f}'.format(test_loss))

def test_loss_test():
    return (test_loss < 0.17)

runtest(test_loss_test, 'test_loss_test')


square_loss(np.mean(ytrain),gt)

ytrain=(ytrain>0)*1.0
ytrain

