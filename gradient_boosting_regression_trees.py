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
from helper import *
%matplotlib notebook

print('You\'re running python %s' % sys.version.split(' ')[0])


xTrSpiral,yTrSpiral,xTeSpiral,yTeSpiral= spiraldata(150)
xTrIon,yTrIon,xTeIon,yTeIon= iondata()


# Create a regression tree with depth 4
tree = RegressionTree(depth=4)

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

def evalboostforest_test0():
    m = 200
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    trees = forest(x, y, m) # create a list of trees 
    preds = evalboostforest(trees, x)
    return preds.shape == y.shape

def evalboostforest_test1():
    m = 200
    x = np.random.rand(10,3)
    y = np.ones(10)
    x2 = np.random.rand(10,3)

    max_depth = 0
    
    # Create a forest with trees depth 0
    # Since the data are all ones, each tree will return 1 as prediction
    trees = forest(x, y, m, max_depth) # create a list of trees      
    pred = evalboostforest(trees, x2)[0]
    return np.isclose(pred,1)  # the prediction should be equal to the sum of weights

def evalboostforest_test2(): 
    # results should match evalforest if alphas are 1/m and labels +1, -1
    m = 20
    x = np.arange(100).reshape((100, 1))
    y = np.sign(np.arange(100))
    trees = forest(x, y, m) # create a list of m trees 

    alphas=np.ones(m)/m
    preds1 = evalforest(trees, x) #evaluate the forest using our own implementation
    preds2 = evalboostforest(trees, x, alphas)
    return np.all(np.isclose(preds1,preds2))

def evalboostforest_test3(): 
    # if only alpha[i]=1 and all others are 0, the result should match exactly 
    # the predictions of the ith tree
    m = 20
    x = np.random.rand(100,5)
    y = np.arange(100)
    x2 = np.random.rand(20,5)

    trees = forest(x, y, m)  # create a list of m trees
    allmatch=True
    for i in range(m): # go through each tree i
        alphas=np.zeros(m)
        alphas[i]=1.0; # set only alpha[i]=1 all other alpha=0
        preds1 = trees[i].predict(x2) # get prediction of ith tree
        preds2 = evalboostforest(trees, x2, alphas) # get prediction of weighted ensemble
        allmatch=allmatch and all(np.isclose(preds1,preds2))
    return allmatch


# and some new tests to check if the weights (and the np.sign function) were incorporated correctly 
runtest(evalboostforest_test0, 'evalboostforest_test0')
runtest(evalboostforest_test1, 'evalboostforest_test1')
runtest(evalboostforest_test2, 'evalboostforest_test2')
runtest(evalboostforest_test3, 'evalboostforest_test3')



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

trees, weights = GBRT(xTrSpiral,yTrSpiral, 50)

def GBRT_test1():
    m = 40
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    trees, weights = GBRT(x, y, m, alpha=0.1)
    return len(trees) == m and len(weights) == m # make sure there are m trees in the forest

def GBRT_test2():
    m = 20
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    max_depth = 4
    trees, weights = GBRT(x, y, m, max_depth)
    depths_forest = np.array([tree.depth for tree in trees]) # Get the depth of all trees in the forest
    return np.all(depths_forest == max_depth) # make sure that the max depth of all the trees is correct

def GBRT_test3():
    m = 4
    xTrSpiral,yTrSpiral,_,_= spiraldata(150)
    max_depth = 4
    trees, weights = GBRT(xTrSpiral, yTrSpiral, m, max_depth, 1) # Create a gradient boosted forest with 4 trees
    
    errs = [] 
    for i in range(m):
        predH = evalboostforest(trees[:i+1], xTrSpiral, weights[:i+1]) # calculate the prediction of the first i-th tree
        err = np.mean(np.sign(predH) != yTrSpiral) # calculate the error of the first i-th tree
        errs.append(err) # keep track of the error
    
    # your errs should be decreasing, i.e., the different between two subsequent errors is <= 0
    return np.all(np.diff(errs) <= 0) 
    
runtest(GBRT_test1, 'GBRT_test1')
runtest(GBRT_test2, 'GBRT_test2')
runtest(GBRT_test3, 'GBRT_test3')

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


def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain,yTrain,w,b,M,Q,trees,weights
    
    if event.key == 'shift': Q+=10
    else: Q+=1
    Q=min(Q,M)
​
    classvals = np.unique(yTrain)
        
    # return 300 evenly spaced numbers over this interval
    res=300
    xrange = np.linspace(0, 1,res)
    yrange = np.linspace(0, 1,res)
    
    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T
​
    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T
​
    # get forest
    
    fun = lambda X:evalboostforest(trees[:Q],X, weights[:Q])        
    # test all of these points on the grid
    testpreds = fun(xTe)
    trerr=np.mean(np.sign(fun(xTrain))==np.sign(yTrain))
    
    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    
    plt.cla()    
    plt.xlim((0,1))
    plt.ylim((0,1))
    # fill in the contours for these predictions
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)
    
    for idx, c in enumerate(classvals):
        plt.scatter(xTrain[yTrain == c,0],xTrain[yTrain == c,1],marker=marker_symbols[idx],color='k')
    plt.show()
    plt.title('# Trees: %i Training Accuracy: %2.2f' % (Q,trerr))
    
        
xTrain=xTrSpiral.copy()/14+0.5
yTrain=yTrSpiral.copy()
yTrain=yTrain.astype(int)
​
# Hyper-parameters (feel free to play with them)
M=50
alpha=0.05;
depth=5;
trees, weights=GBRT(xTrain, yTrain, M,alpha=alpha,maxdepth=depth)
Q=0;
​
​
%matplotlib notebook
fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick_forest) 
print('Click to add a tree.');
plt.title('Click to start boosting on the spiral data.')
visclassifier(lambda X: np.sum(X,1)*0,xTrain,yTrain,newfig=False)
plt.xlim(0,1)
plt.ylim(0,1)



def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain,yTrain,Q,trees,weights
    
    if event.key == 'shift': Q+=10
    else: Q+=1
    Q=min(Q,M)


    plt.cla()    
    plt.xlim((0,1))
    plt.ylim((0,1))
    pTest=evalboostforest(trees[:Q],xTest,weights[:Q])    
    pTrain=evalboostforest(trees[:Q],xTrain,weights[:Q])    


    errTrain=np.sqrt(np.mean((pTrain-yTrain)**2))
    errTest=np.sqrt(np.mean((pTest-yTest)**2))

    plt.plot(xTrain[:,0],yTrain,'bx')
    plt.plot(xTest[:,0],pTest,'r-')
    plt.plot(xTest[:,0],yTest,'k-')

    plt.legend(['Training data','Prediction'])
    plt.title('(%i Trees)  Error Tr: %2.4f, Te:%2.4f' % (Q,errTrain,errTest))
    plt.show()
    
        
n=100;
NOISE=0.05
xTrain=np.array([np.linspace(0,1,n),np.zeros(n)]).T
yTrain=2*np.sin(xTrain[:,0]*3)*(xTrain[:,0]**2)
yTrain+=np.random.randn(yTrain.size)*NOISE;
ntest=300;
xTest=np.array([linspace(0,1,ntest),np.zeros(ntest)]).T
yTest=2*np.sin(xTest[:,0]*3)*(xTest[:,0]**2)



    
# Hyper-parameters (feel free to play with them)
M=400
alpha=0.05;
depth=3;
trees, weights=GBRT(xTrain, yTrain, M,alpha=alpha,maxdepth=depth)
Q=0;

%matplotlib notebook
fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick_forest) 
print('Click to add a tree (shift-click and add 10 trees).');
plt.title('Click to start boosting on this 1D data (Shift-click to add 10 trees).')
plt.plot(xTrain[:,0],yTrain,'*')
plt.plot(xTest[:,0],yTest,'k-')
plt.xlim(0,1)
plt.ylim(0,1)


def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain,yTrain,w,b,M
    # create position vector for new point
    pos=np.array([[event.xdata,event.ydata]]) 
    if event.key == 'shift': # add positive point
        color='or'
        label=1
    else: # add negative point
        color='ob'
        label=-1    
    xTrain = np.concatenate((xTrain,pos), axis = 0)
    yTrain = np.append(yTrain, label)
    marker_symbols = ['o', 'x']
    classvals = np.unique(yTrain)
        
    w = np.array(w).flatten()
    
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    
    # return 300 evenly spaced numbers over this interval
    res=300
    xrange = np.linspace(0, 1,res)
    yrange = np.linspace(0, 1,res)
    
    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # get forest
    trees, weights=GBRT(xTrain, yTrain, M)
    fun = lambda X:evalboostforest(trees,X, weights)
    # test all of these points on the grid
    testpreds = fun(xTe)
    
    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly
    
    plt.cla()    
    plt.xlim((0,1))
    plt.ylim((0,1))
    # fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)
    
    for idx, c in enumerate(classvals):
        plt.scatter(xTrain[yTrain == c,0],
            xTrain[yTrain == c,1],
            marker=marker_symbols[idx],
            color='k'
            )
    plt.show()
    
        
xTrain= np.array([[5,6]])
b=yTrIon
yTrain = np.array([1])
w=xTrIon
M=5

%matplotlib notebook
fig = plt.figure()
plt.xlim(0,1)
plt.ylim(0,1)
cid = fig.canvas.mpl_connect('button_press_event', onclick_forest)
print('Note: You may notice a delay when adding points to the visualization.')
plt.title('Use shift-click to add negative points.')


