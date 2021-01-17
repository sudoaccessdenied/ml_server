# -*- coding: utf-8 -*-

"""
Created on Sun Jan  3 16:16:35 2021

@author: Nishant
"""

import numpy as np
import json
# import pandas as pd
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import h5py
# import scipy
# from PIL import Image
# from scipy import ndimage

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def evaluate(y,yhat):
    y =y.astype('int32') 
    yhat = yhat.astype('int32')
    matrix_test = confusion_matrix(y, yhat)
    
    TP = matrix_test[0][0]
    FN = matrix_test[0][1]
    FP = matrix_test[1][0]
    TN = matrix_test[1][1]

    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    accuracy = (TP+TN)/(TP+TN+FN+FP)


    result = {
            "senstivity":sensitivity,
            "specificity":specificity,
            "accuracy":accuracy*100
            }
    return result    
    
    


def sigmoid(z):
    exponent = 1/np.exp(z)
    
    s = 1/(1+exponent)
    
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w, b

def propagate(w, b, X, Y):    
    m = X.shape[1]
    z = np.dot(w.T,X)+b
    A = sigmoid(z)                                    
    cost = -np.sum(Y*np.log(A) +(1-Y)*np.log(1-A))    
    cost/=m
    dz = A-Y
    dw = np.dot(X,dz.T)
    db = np.sum(dz)
    dw/=m
    db/=m   
    cost = np.squeeze(cost)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):
    
    costs = []
    
    for i in range(num_iterations):  
        grads, cost = propagate(w,b,X,Y)
  
        dw = grads["dw"]
        db = grads["db"]
  
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
  
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
  
    z = np.dot(w.T,X)+b
    A = sigmoid(z)
    
    for i in range(A.shape[1]):
        
          Y_prediction[0,i] = 1 if (A[0,i]>0.5) else 0
    
    
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
    m = X_train.shape[0]
    w, b = initialize_with_zeros(m)
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate)
    
    
    w = parameters["w"]
    b = parameters["b"]
    
    
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    # print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    # print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    # trainA = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    # testA = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    costs = json.dumps(costs, cls =NumpyEncoder)
    # # matrix_train = confusion_matrix(Y_prediction_train,Y_train)
    Y_prediction_test = Y_prediction_test.reshape((X_test.shape[1],1))
    Y_test = Y_test.reshape((X_test.shape[1],1))

    Y_prediction_train= Y_prediction_train.reshape((X_train.shape[1],1))
    Y_train = Y_train.reshape((X_train.shape[1],1))


    # print(Y_prediction_test[:5])
    # print("Evaluate")
    # print(Y_test[:5])
   
    d = {
        "costs": costs,
          "test_accuracy": evaluate(Y_test,Y_prediction_test), 
           "train_accuracy" : evaluate(Y_train,Y_prediction_train), 
          "error_type":"Cross Entropy",
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations
         }

    return d