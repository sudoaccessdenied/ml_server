# -*- coding: utf-8 -*-

import numpy as np
import math
import operator
from sklearn.metrics import confusion_matrix
# from algorithms.loadDataset import loadDataset


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
    


def euclideanDistance(instance1, instance2,length):
    distance = 0
    for x in range(length):
        distance+=pow((instance1[x]-instance2[x]),2)
#         print(distance)
    return math.sqrt(distance);

def getNeighbour(trainSet,testInstance,k):
    distance = []
    length = len(testInstance)-1
    for x in range(len(trainSet)):
        dist = euclideanDistance(testInstance,trainSet[x],length)
        distance.append((trainSet[x],dist))
    
    distance.sort(key = operator.itemgetter(1))
    neighbour = []
    for x in range(k):
        neighbour.append(distance[x][0])
    
    return neighbour

def getResponse(neighbour):
    classVotes ={}
    for x in range(len(neighbour)):
        response = neighbour[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    
    sortedVotes = sorted(classVotes.items(),key = operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

def accuracy(testSet, predicted):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] in predicted[x]:
            correct+=1;
    return (correct/float(len(testSet)))*100



def model(X_train, Y_train, X_test, Y_test,k = 4):

    prediction=[]
    neighbours=[]
    
    train_set = np.concatenate((X_train, Y_train),axis=1)
    test_set = np.concatenate((X_test, Y_test),axis=1)
    m =len(X_test)
    for x in range(len(test_set)):
        neighbours = getNeighbour(train_set,test_set[x],k)
        result = getResponse(neighbours)
        prediction.append(result)
        
    
    prediction = np.array(prediction,dtype=np.int32)
    
    Y_test = Y_test.astype('int32')
    Y_test = Y_test.flatten()
    
    d = {
          "test_accuracy": evaluate(np.array(Y_test),np.array(prediction)), 
          # "train-accuracy" : np.sum((prediction == Y_test)/m), 
         #  "error-type":"Cross Entropy",
         # "learning_rate" : learning_rate,
         # "num_iterations": num_iterations
         }

    return d

