#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:35 2021

@author: Nishant
"""
# import numpy as np
from flask import Flask, request, jsonify ,url_for
from algorithms.loadDataset import loadDataset
from algorithms import linearRegression as lr
from algorithms import logisticRegression as lg
from algorithms import neuralnetworks as nn
from algorithms import knn as knn
from algorithms import kmeans as km
from flask_cors import  cross_origin

app = Flask(__name__)

@app.route('/linear_reg_api',methods=['POST'])
@cross_origin()
def linear_reg_api():
    '''
    Linear regression algo
    '''
    data = request.get_json(force=True)
    X_train, X_test, y_train, y_test = loadDataset('reg',data['select'],data['split'])
    d = lr.model(X_train.T,y_train.T,X_test.T,y_test.T,data['learning_rate'],data['num_iterations'])
    return jsonify(d) 

@app.route('/logistic_reg_api',methods=['POST'])
@cross_origin()
def logistic_reg_api():
    '''
    Logistic regression algo
    '''
    data = request.get_json(force=True)
    X_train, X_test, y_train, y_test = loadDataset('class',data['select'],data['split'])
    d = lg.model(X_train.T,y_train.T,X_test.T,y_test.T,data['num_iterations'],data['learning_rate'])
    return jsonify(d) 


@app.route('/neural_network_api',methods=['POST'])
@cross_origin()
def neural_network_api():
    '''
    Neural Network algo
    '''
    data = request.get_json(force=True)
    X_train, X_test, y_train, y_test = loadDataset('class',data['select'],data['split'])
    
    layers_dims = [X_train.shape[1]]
    layers_dims.extend(data['hidden_layer'])
    layers_dims.append(1)
    
    d = nn.L_layer_model(X_train.T,y_train.T,X_test.T,y_test.T,layers_dims,data['learning_rate'],data['num_iterations'])
    return jsonify(d) 



@app.route('/knn_api',methods=['POST'])
@cross_origin()
def knn_api():
    '''
    KNN algo
    '''
    data = request.get_json(force=True)
    X_train, X_test, y_train, y_test = loadDataset('class',data['select'],data['split'])
    
    
    d = knn.model(X_train,y_train,X_test,y_test,data['k'])
    return jsonify(d) 

@app.route('/kmeans_api',methods=['POST'])
@cross_origin()
def kmeans_api():
    '''
    KNN algo
    '''
    data = request.get_json(force=True)
    X = km.load_dataset()
    
    
    d = km.model(X,data['k'],data['dist_type'],data['num_iterations'])
    d = url_for('static', filename='kmeans.png')
    return jsonify(d) 
















if __name__ == "__main__":
    app.run(debug=True)