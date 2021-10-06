# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:19:20 2020

@author: ecornell 
Decision Trees and Model Selection
Cross_validation

"""
import numpy as np
from pylab import *
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time


data = loadmat("ion.mat")
xTr  = data['xTr'].T
yTr  = data['yTr'].flatten()
xTe  = data['xTe'].T
yTe  = data['yTe'].flatten()

# Create a regression tree with no restriction on its depth. 
# This is equivalent to what you implemented in the previous project
# if you want to create a tree of max depth k
# then call h.RegressionTree(depth=k)
tree = RegressionTree(depth=np.inf)

# To fit/train the regression tree
tree.fit(xTr, yTr)

# To use the trained regression tree to make prediction
pred = tree.predict(xTr)

def square_loss(pred, truth):
    return np.mean((pred - truth)**2)

print('Training Loss: {:.4f}'.format(square_loss(tree.predict(xTr), yTr)))
print('Test Loss: {:.4f}'.format(square_loss(tree.predict(xTe), yTe)))


def grid_search(xTr, yTr, xVal, yVal, depths):
    '''
    Input:
        xTr: nxd matrix
        yTr: n vector
        xVal: mxd matrix
        yVal: m vector
        depths: a list of len k
    Return:
        best_depth: the depth that yields that lowest loss on the validation set
        training losses: a list of len k. the i-th entry corresponds to the the training loss
                the tree of depths[i]
        validation_losses: a list of len k. the i-th entry corresponds to the the validation loss
                the tree of depths[i]
    
    '''
    k = len(depths)
    training_losses = np.zeros(k)
    validation_losses = np.zeros(k)
    best_depth = None
    #k = len(depths)
    for i in range(k):
        tree = RegressionTree(depth=depths[i])
        tree.fit(xTr, yTr)
        train_pred = tree.predict(xTr)
        training_losses[i]=square_loss(tree.predict(xTr), yTr)
        valid_pred=tree.predict(xVal)
        validation_losses[i]=np.mean((valid_pred - yVal)**2)
        
    loss=validation_losses[0] 
    index=0
    for i in range(k):
        if validation_losses[i]<loss:
            loss= validation_losses[i]
            index=i
            
    best_depth=depths[index]
    
    return best_depth, training_losses, validation_losses


def generate_kFold(n, k):
    '''
    Input:
        n: number of training examples
        k: number of folds
    Returns:
        kfold_indices: a list of len k. Each entry takes the form
        (training indices, validation indices)
        Part 2 : Create k fold index from input data (generate_kFold)
1. Set n & k.  Here n is the sample size and k is fold size (number of cross - validation). 
2. Set the index of total record using n [1,2,3,,,,,n]
3. Find the folder size ( n// k). The folder size is the number of records in each fold. 
4. Divide the index of total record with folder size like [1,2], [3,4],,, when  n = 6 and k = 3
5. Create the index of training and test data like [[1,2], [3,4]] / [5,6], [[3,4], [5,6]/[1,2], [ [5,6],[1,2]]/ [3,4]


    '''
    assert k >= 2
    kfold_indices = []
    index=np.arange(n)
    #train_index=[]
    
    #train_index=[]
    #train_index=np.zeros(k)
    #val_index=np.zeros(k)
    folder_index=[]
    folder_size=int(n/k)
    #folder_index=[index[i*folder_size:(i+1)*folder_size] for i in range(k-1)]
    for i in range(k-1):
        folder_index.append(index[i*folder_size:(i+1)*folder_size])
    folder_index.append(index[(k-1)*folder_size:])
    for i in range(k):
        train_index=[folder_index[iteration] for iteration in range(k) if iteration !=i]
        val_index=folder_index[i]
       
                
        kfold_indices.append((np.concatenate(train_index), val_index))
     
    return  kfold_indices

generate_kFold(3,3)

def cross_validation(xTr, yTr, depths, indices):
    '''
    Input:
        xTr: nxd matrix (training data)
        yTr: n vector (training data)
        depths: a list (of length l) depths to be tried out
        indices: indices from generate_kFold
    Returns:
        best_depth: the best parameter 
        training losses: a list of lenth l. the i-th entry corresponds to the the average training loss
                the tree of depths[i]
        validation_losses: a list of length l. the i-th entry corresponds to the the average validation loss
                the tree of depths[i] 
                
                
Part 3 : Find the best depth of tree using cross validation and grid search (cross_validation)
1. Prepare training and validation data using k-fold index
2. Calculate the loss of Regression Tree at depth of tree [1,2,3,4] using the training and validation data
3. Take the mean of loss of training & validation data at each depth of tree
4. Find the depth of tree of lowest loss of validation data (This is our depth of tree)
    '''
    training_losses = []
    validation_losses = []
    best_depth = None
    for i, j in indices:
        x_train, y_train= xTr[i], yTr[i]
        x_val, y_val=xTr[j], yTr[j]
        best_depth, training_losses_it, validation_losses_it=grid_search(x_train, y_train, x_val, y_val, depths)
        training_losses.append(training_losses_it)
        validation_losses.append(validation_losses_it)
        
        
        
    training_losses=np.mean(training_losses, axis=0)
    validation_losses=np.mean(validation_losses, axis=0)
    best_depth=depths[np.argmin(validation_losses)]
    
    return best_depth, training_losses, validation_losses



