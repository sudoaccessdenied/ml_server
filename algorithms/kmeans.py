# -*- coding: utf-8 -*-

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import csv
import json
import os


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def initializeCentroid(X,K):
    row, col = X.shape
    randIdx = np.random.permutation(range(row))
    centroids = []
    for val in randIdx[1:K+1]:
        centroids.append(X[val]) 
    return np.array(centroids)

def euclideanDistance(X,Y):
    diff = X-Y
    diff = np.power(diff,2)
    dist = np.sqrt(np.sum(diff,axis=0))
    return dist


def manhattanDistance(X,Y):
    diff = np.abs(X-Y)
    dist = np.sum(diff,axis=0)
    return dist
    
def findCentroid(X,centroids,distType ='euc'):
    idx =[]
    
    for i,example in enumerate(X):
        tempDistance = {}
        for index ,centroid in enumerate(centroids):
            if distType =='man':
                tempDistance[index] = manhattanDistance(example,centroid)
            else:
                tempDistance[index] = euclideanDistance(example,centroid)
        
        tempDistance = {k: v for k, v in sorted(tempDistance.items(), key=lambda item: item[1])}
        
        idx.append(next(iter(tempDistance))) 

    return idx

def findClusterMean(X,centroids,idx,K):
    newCentroids = np.zeros(centroids.shape)
    freq = np.zeros(K)
    for index, val in enumerate(idx):
        newCentroids[val]+=X[index]
        freq[val]=freq[val]+1
    
    for index, row in enumerate(newCentroids):
        if(freq[index]>0):
            newCentroids[index] = newCentroids[index]/freq[index]
    
    return newCentroids

def computeCost(X,centroids,idx):
    m = X.shape[0]
    diff = np.zeros(X.shape)
    for i , val in enumerate(X):
        diff[i] = val - centroids[idx[i]]
    diff = np.power(diff,2)
    dist = np.sum(diff)/m
    return dist

def model(X,K,distType,numIter= 5):
    costs =[]
    centroids =initializeCentroid(X,K)

    for i in range(numIter):
        idx = findCentroid(X,centroids,distType)
        centroids = findClusterMean(X,centroids,idx,K)
        cost = computeCost(X,centroids,idx)
        costs.append(cost)
        
    # costs = json.dumps(costs, cls =NumpyEncoder)
    # X = json.dumps(X, cls =NumpyEncoder)
    # idx = json?.dumps(idx, cls =NumpyEncoder)
    # centroids = json.dumps(centroids, cls =NumpyEncoder)
    
    
    
    color = ["blue","green","red","yellow","pink","brown","orange","purple"]
    # print(X)
    # idx = d['idx']
    
    c = [color[i]  for i in idx]
    
    
    # print(idx)
    if os.path.isfile('static/kmeans.png'):
        os.unlink("static/kmeans.png")

    plt.scatter(X[:,1],X[:,0],color=c)
    plt.draw()
    fig1 = plt.gcf()
    fig1.savefig("static/kmeans.png")
    plt.close()
    
    return {"done":True}


def load_dataset():
        X= -2 * np.random.rand(150,3)
        X1 = 1 + 2 * np.random.rand(50,3)
        X2 = 4+ 2 * np.random.rand(50,3)
        X[50:100, :] = X1
        X[100:150, :] = X2
        return X

# X = np.random.randint(10,size=(10,2))


# X = []
# load_dataset("./datasets/student.csv",X)
# X = np.array(X)
# print(X)
# K=3
# randIdx = np.random.permutation(range(5))

# X = load_dataset()
# K = 3
# d = model(X,K,10,1)
# print(d)