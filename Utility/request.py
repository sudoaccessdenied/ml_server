# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:35 2021

@author: Nishant
"""
    

import requests


# url = 'http://localhost:5000/linear_reg_api'
data = {'num_iterations':10, 
        'select':2, 
        'learning_rate':.0075,
        'dist_type':'man',
        'split':.2,
        'hidden_layer':[20,7,5,3],
        'k':3
        }

# r = requests.post(url,json=data)

# print(r.json())



# url1 = 'http://localhost:5000/logistic_reg_api'

# logistic =  requests.post(url1,json=data)
# print(logistic.json())



# url2 = 'http://localhost:5000/neural_network_api'

# nn =  requests.post(url2,json=data)
# print(nn.json())




# url3 = 'http://localhost:5000/knn_api'

# knn =  requests.post(url3,json=data)
# print(knn.json())



# url3 = 'http://localhost:5000/kmeans_api'

# kmeans =  requests.post(url3,json=data)
# print(kmeans.json())


