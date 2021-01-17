# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:35 2021

@author: Nishant
"""
from sklearn import datasets as d
from sklearn.model_selection import train_test_split
import numpy as np
# import pandas as pd
import h5py

from sklearn.preprocessing import normalize



def loadDataset(algoType,dataset,split):
    
    if algoType =='reg':
        if dataset == 1:
            X,Y = d.load_boston(return_X_y=True)
            
        else:
            X,Y = d.load_diabetes(return_X_y=True)
    elif algoType =='class':
        if dataset == 1:
            X,Y = d.load_breast_cancer(return_X_y=True)
        else :
            return load_catvnocat()
    
        
    X = normalize(X)

    Y = Y.reshape((X.shape[0],1))
      # print(Y.shape)
    # Y = normalize(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split, random_state=42)
        

    
    return X_train,X_test,y_train,y_test

# X_train, X_test, y_train, y_test = loadDataset('reg',2,.33)
               

def load_catvnocat():
    with h5py.File('dataset/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('dataset/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        # classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0],1 ))
    test_set_y_orig = test_set_y_orig.reshape(( test_set_y_orig.shape[0],1 ))

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1)
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1)
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.
    
    return train_set_x, test_set_x, train_set_y_orig , test_set_y_orig

load_catvnocat()
