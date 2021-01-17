# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:35 2021

@author: Nishant
"""


import numpy as np
# from algorithms.loadDataset import loadDataset
from flask import jsonify
import json
# from sklearn.metrics import r2_score

# coefficient_of_dermination = r2_score(y, p(x))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def initialize(x):
    n = x.shape[0]
    w = np.zeros((n,1))
    b = 1
    return w,b


def linear_regresstion(x,y,w,b,alpha,no_of_iteration):
    costs=[]
    m = x.shape[1]
    for i in range(1,no_of_iteration+1):
        z = np.dot(w.T,x)
        cost = np.sum(np.square(z-y)*0.5)
        dz = z-y
        dw = np.dot(x,dz.T)
        db = np.sum(dz)
        dw/=2*m
        db/=2*m
        w = w-alpha*dw
        b=b-alpha*db
        if i%100==0:
            costs.append(cost)
        
    return w,b,costs


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    Y_prediction = np.dot(w.T,X)+b  
#     assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def r2Score(y,yhat):
    ybar = np.mean(y)         
    rss = np.sum((y-yhat)**2)   
    tss = np.sum((y - ybar)**2)    
    return 1-(rss /tss)


def mse(y,yhat):
    error = np.mean((y-yhat)**2) 
    return error


def model(x_train,y_train,x_test,y_test,alpha,no_of_iteration):
    w,b = initialize(x_train)
    
    w,b,costs = linear_regresstion(x_train,y_train,w,b,alpha,no_of_iteration)
    costs = np.array(costs,dtype =np.float64)
    Y_prediction_train = predict(w,b,x_train)
    Y_prediction_test = predict(w,b,x_test)
    
    # trainA = np.mean(np.abs((Y_prediction_train - y_train)/y_train))
    # testA =    np.mean(np.abs((Y_prediction_test - y_test)/y_test))
    
  
    trainA = mse(y_train, Y_prediction_train)
    testA = mse(y_test, Y_prediction_test)
    # print(Y_prediction_test[:5])
    
    costs = json.dumps(costs, cls =NumpyEncoder)
    
    
    
    d = {
        "costs": costs,
         "test_accuracy": testA, 
         "train_accuracy" : trainA, 
         "error-type":"MSE",
         # "w" : w, 
         # "b" : b,
         "learning_rate" : alpha,
         "num_iterations": no_of_iteration
         }

    return d
    

# X_train, X_test, y_train, y_test = loadDataset('reg',1,.33)
               
# print(X_train.shape)
# y_train = 
# d = model(X_train.T,y_train.T,X_test.T,y_test.T,.8,1500)
    
# print(d)
