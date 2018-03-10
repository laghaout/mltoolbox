#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:36:24 2018

@author: Amine Laghaout
"""
  
import classifiers as cla
import data_wranglers as dat

problem = 'iris'

#%% Data object

data_object_train = False
data_object_test = False

if problem in {'iris', 'digits'}:

    data_object_train = dat.sklearn_dataset(problem)
    
elif problem == 'imdb':
    
    data_object_train = dat.keras_imdb(
            params = {'top_words': 5000, 'max_review_length': 500})
    
#%% Classifier object

if problem == 'iris':

    from pandas import DataFrame
    classifier_object = cla.MLP(
            params = {'input_dim': data_object_train.data.input.shape[1], 
                      'output_dim': data_object_train.data.output.shape[1], 
                      'epochs': 10, 
                      'batch_size': None, 
                      'verbose': 0, 
                      'validation_split': 1/3, 
                      'loss_function': 'categorical_crossentropy', 
                      'architecture': DataFrame(
                              {'num_nodes': [10, 8, 6, 3], 
                               'activation': ['tanh', 'tanh', 'tanh', 'softmax'], 
                               'dropout': 0
                               }),  
                      'optimizer': {'lr': 0.01, 
                                    'decay': 1e-6, 
                                    'momentum': 0.9, 
                                    'nesterov': True}, 
                      'metrics': ['accuracy']})
    
elif problem == 'digits': # verified
    
    from pandas import DataFrame
    classifier_object = cla.MLP(
            params = {'input_dim': data_object_train.data.input.shape[1], 
                      'output_dim': data_object_train.data.output.shape[1], 
                      'epochs': 30, 
                      'batch_size': None, 
                      'verbose': 0, 
                      'validation_split': 1/3, 
                      'loss_function': 'categorical_crossentropy',
                      'metrics': ['accuracy'], 
                      'architecture': DataFrame(
                              {'num_nodes': [64, 64, None], 
                               'activation': ['relu', 'relu', 'softmax'], 
                               'dropout': 0.1}), 
                      'optimizer': {'lr': 0.01, 
                                    'decay': 1e-6, 
                                    'momentum': 0.9, 
                                    'nesterov': True}})    

#%% Machine learning pipeline

return_vars = cla.pipeline(classifier_object, 
                           #examine = data_object_train,
                           train = data_object_train, 
                           test = data_object_test, 
                           params = {'marker': None})

try: data_raw_train = data_object_train.data_raw 
except: pass
try:  data_raw_test = data_object_test.data_raw 
except: pass

