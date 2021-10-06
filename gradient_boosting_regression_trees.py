# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:27:56 2020

@author: Mariam Arzumanyan

Gradient Boosting Regression Trees
"""
import numpy as np
from pylab import *
from numpy.matlib import repmat
import sys
import matplotlib 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time


xTrSpiral,yTrSpiral,xTeSpiral,yTeSpiral= spiraldata(150)
xTrIon,yTrIon,xTeIon,yTeIon= iondata()


# Create a regression tree with depth 6
tree = RegressionTree(depth=6)

# To fit/train the regression tree
tree.fit(xTrSpiral, yTrSpiral)

# To use the trained regression tree to predict a score for the example
score = tree.predict(xTrSpiral)

# To use the trained regression tree to make a +1/-1 prediction
pred = np.sign(tree.predict(xTrSpiral))


# Evaluate the depth 4 decision tree
# tr_err   = np.mean((np.sign(tree.predict(xTrSpiral)) - yTrSpiral)**2)
# te_err   = np.mean((np.sign(tree.predict(xTeSpiral)) - yTeSpiral)**2)

print("Training error: %.4f" % np.mean(np.sign(tree.predict(xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(tree.predict(xTeSpiral)) != yTeSpiral))


def visclassifier(fun,xTr,yTr,newfig=True):
    """
    visualize decision boundary
    Define the symbols and colors we'll use in the plots later
    """

    yTr = np.array(yTr).flatten()
    
    symbols = ["ko","kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    # get the unique values from labels array
    classvals = np.unique(yTr)

    if newfig:
        plt.figure()

    # return 300 evenly spaced numbers over this interval
    res=300
    xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]),res)
    yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]),res)
    
    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    
    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # test all of these points on the grid
    testpreds = fun(xTe)
    
    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly
    
    # fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    # creates x's and o's for training set
    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            color='k'
                   )

    plt.axis('tight')
    # shows figure and blocks
    plt.show()
    

visclassifier(lambda X: tree.predict(X),xTrSpiral,yTrSpiral)


def evalboostforest(trees, X, alphas=None):
    """Evaluates X using trees.
    
    Input:
        trees:  list of TreeNode decision trees of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector
        
    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(trees)
    n,d = X.shape
    
    if alphas is None:
        alphas = np.ones(m) / len(trees)
            
    pred = np.zeros(n)
    for i in range(m):
        pred+=alphas[i]*trees[i].predict(X)
    
    return pred



def GBRT(xTr, yTr, m, maxdepth=4, alpha=0.1):
    """Creates GBRT.
    
    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree
        alpha:    learning rate for the GBRT
        
        
    Output:
        trees: list of decision trees of length m
        weights: weights of each tree
    """
    
    n, d = xTr.shape
    trees = []
    weights = []
    
    # Make a copy of the ground truth label
    # this will be the initial ground truth for our GBRT
    # This should be updated for each iteration
    t = np.copy(yTr)
    for i in range(m):
        tree = RegressionTree(depth=maxdepth)
        tree.fit(xTr, t)
        trees.append(tree)
        t=evalboostforest(trees, xTr, alphas=None)
        weights.append(t)
        t=yTr-t
       
    return trees, weights

trees, weights=GBRT(xTrSpiral,yTrSpiral, 40, maxdepth=4, alpha=0.03) # compute tree on training data 
visclassifier(lambda X:evalboostforest(trees, X, weights),xTrSpiral,yTrSpiral)
print("Training error: %.4f" % np.mean(np.sign(evalforest(trees,xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(evalforest(trees,xTeSpiral)) != yTeSpiral))



M=40 # max number of trees
err_trB=[]
err_teB=[]
alltrees, allweights =GBRT(xTrIon,yTrIon, M, maxdepth=4, alpha=0.05)
for i in range(M):
    trees=alltrees[:i+1]
    weights=allweights[:i+1]
    trErr = np.mean(np.sign(evalboostforest(trees,xTrIon, weights)) != yTrIon)
    teErr = np.mean(np.sign(evalboostforest(trees,xTeIon, weights)) != yTeIon)
    err_trB.append(trErr)
    err_teB.append(teErr)
    print("[%d]training err = %.4f\ttesting err = %.4f" % (i,trErr, teErr))

plt.figure()
line_tr, = plt.plot(range(M), err_trB, '-*', label="Training Error")
line_te, = plt.plot(range(M), err_teB, '-*', label="Testing error")
plt.title("Gradient Boosted Trees")
plt.legend(handles=[line_tr, line_te])
plt.xlabel("# of trees")
plt.ylabel("error")
plt.show()









