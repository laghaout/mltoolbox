#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 08:34:14 2018

@author: Amine Laghaout
"""

class problem:
    
    def __init__(self, default, **kwargs):
        
        # Default arguments        
        [self.__setattr__(k, default[k]) for k in default]
        
#        # Arguments specified explicitly. These overwrite the default 
#        # arguments.
        [self.__setattr__(k, kwargs[k]) for k in kwargs]
        
        # TODO: Search all the dictionary keys and overwrite if applicable.
        
        self.data_object_train = False
        self.data_object_test = False
        self.data_object_serve = False
        
    def pipeline(self, examine = False, select = False, train = False, 
                 test = False, serve = False, model_summary = False, 
                 params = {'marker': None}):
        
        """
        Parameters
        ----------
        
        classifier_object : classifier
            Instance of a classifier object ``classifiers.classifier()``
        examine, select, train, test, serve : data_wrangler
            Instances of a data wrangler object ``data_wranglers.data_wrangler()``
            used for data examination, model selection, training, testing, and 
            serving
        params : dict
            Miscellaneous parameters
        
        Returns
        -------
        
        self.results : dict
            Dictionary of evaluations
        """
        
        from collections import namedtuple
        
        self.results = {x: dict() for x in ('select', 'train', 'test', 'serve')}
        params = namedtuple('params', params.keys())(**params)    
    
        print('\n**********************************************************************')    
        print('*** Executing the pipeline for ', self.name, '.', sep = '')
        print('**********************************************************************')    
        
    
        if model_summary:
    
            print('\n******************** MODEL SUMMARY:')
            self.classifier_object.model.summary()
        
        if examine is not False:
            
            print('\n******************** EXAMINING:')
            self.results['train']['examination'] = self.data_object_train.examine()
    
        if select is not False:
            
            print('\n******************** SELECTING:')
            self.classifier_object.select(self.data_object_train)
        
        if train is not False:
            
            print('\n******************** TRAINING:')
            self.classifier_object.train(self.data_object_train)
            self.classifier_object.train_report(marker = params.marker)
            self.results['train']['prediction'] = self.classifier_object.test(self.data_object_train)
            
        if test is not False: 
            
            print('\n******************** TESTING:')        
            self.results['test']['prediction'] = self.classifier_object.test(self.data_object_test)
    
        if serve is not False: 
            
            print('\n******************** SERVING:')
            self.results['serve']['prediction'] = self.classifier_object.test(self.data_object_serve)
            
        return self.results
    

class random_walk(problem): 
    
    def __init__(
            self,
            default = {
                    'name': 'random walks',
                    'nrows': 10, 
                    'input_dim': 2, 
                    'output_dim': 1, 
                    'min_seq_len': 2, 
                    'max_seq_len': 20, 
                    'max_step': 1, 
                    'validation_split': 1/3,
                    'metrics': ['accuracy'], 
                    'params': {'embed_dim': 39, 
                               'num_nodes': 20, 
                               'dropout': 0.2, 
                               'recurrent_dropout': 0.2, 
                               'epochs': 11, 
                               'batch_size': 128, 
                               'activation': 'sigmoid', 
                               'loss_function': 'binary_crossentropy', 
                               'optimizer': 'adam'},
                    'verbose': 1},
            **kwargs):
        
        super().__init__(default, **kwargs)
        
        import classifiers as cla
        import data_wranglers as dat
        
        self.data_object_train = dat.random_walk(
                nrows = self.nrows, 
                input_dim = self.input_dim, 
                output_dim = self.output_dim, 
                min_seq_len = self.min_seq_len,
                max_seq_len = self.max_seq_len, 
                max_step = self.max_step,
                verbose = self.verbose)
        
        self.data_object_test = dat.random_walk(
                nrows = self.nrows, 
                input_dim = self.input_dim, 
                output_dim = self.output_dim, 
                min_seq_len = self.min_seq_len,
                max_seq_len = self.max_seq_len, 
                max_step = self.max_step,
                verbose = self.verbose)        

        self.data_object_test.plot()
        
        self.classifier_object = cla.RNN2(
                input_dim = self.input_dim,
                output_dim = self.data_object_train.data.output.shape[1], 
                max_seq_len = max(self.data_object_train.data.input.shape[1],
                                  self.data_object_test.data.input.shape[1]),
                verbose = self.verbose,
                validation_split = self.validation_split, 
                metrics = self.metrics,
                params = self.params)        
                
class dummy(problem): 
    
    def __init__(self, 
                 default = {'name': 'dummy problem',
                            'verbose': 0, 
                            'algo': 'MLP', 
                            'planet': 'Earth', 
                            'acceleration': 9.80665,
                            'params': {'hyperparam_1': 'A',
                                       'hyperparam_2': 60.5}}, 
                 **kwargs):
        
        super().__init__(default, **kwargs)
        
        import classifiers as cla
        import data_wranglers as dat
        
        self.data_object_train = dat.dummy()
        self.data_object_test = dat.dummy()
        self.data_object_serve = dat.dummy()        
        
        self.classifier_object = cla.dummy(params = self.params)

class iris(problem): 
    
    from pandas import DataFrame
    
    def __init__(
            self,
            default = {
                    'name': 'iris (scikit-learn)',
                    'verbose': 1,
                    'validation_split': 1/3, 
                    'metrics': ['accuracy'],
                    'params': {'epochs': 40, 
                               'batch_size': 1, 
                               'loss_function': 'categorical_crossentropy',  
                               'optimizer': 'adam', #
                               'architecture': DataFrame(
                                       {'num_nodes': [16, None], 
                                        'activation': ['sigmoid', 'softmax'], 
                                        'dropout': 0.0})}},
                 **kwargs):
    
        super().__init__(default, **kwargs)
        
        import classifiers as cla
        import data_wranglers as dat
        
        self.data_object_train = dat.sklearn_dataset('iris')
        self.data_object_test = dat.sklearn_dataset('iris')
        
        self.classifier_object = cla.MLP(
                input_dim = self.data_object_train.data.input.shape[1], 
                output_dim = self.data_object_train.data.output.shape[1],  
                validation_split = self.validation_split, 
                verbose = self.verbose,
                metrics = self.metrics, 
                params = self.params)
        
class digits(problem):
    
    from pandas import DataFrame
    
    def __init__(
            self, 
            default = {
                    'name': 'digits (scikit-learn)', 
                    'validation_split': 1/3, 
                    'verbose': 0, 
                    'metrics': ['accuracy'],  
                    'params': {'epochs': 30, 
                               'batch_size': None, 
                               'loss_function': 'categorical_crossentropy', 
                               'optimizer': {'lr': 0.01, 
                                             'decay': 1e-6, 
                                             'momentum': 0.9, 
                                             'nesterov': True}, 
                               'architecture': DataFrame(
                                       {'num_nodes': [64, 30, None], 
                                        'activation': ['sigmoid', 'sigmoid', 'softmax'], 
                                        'dropout': 0.1})}}, 
                 **kwargs):
        
        super().__init__(default, **kwargs)

        import classifiers as cla
        import data_wranglers as dat
        
        self.data_object_train = dat.sklearn_dataset('digits')
        self.data_object_test = dat.sklearn_dataset('digits')
        
        self.classifier_object = cla.MLP(
                input_dim = self.data_object_train.data.input.shape[1],
                output_dim = self.data_object_train.data.output.shape[1],
                validation_split = self.validation_split, 
                verbose = self.verbose, 
                metrics = self.metrics, 
                params = self.params)    

class imdb(problem): 

    def __init__(
            self,
            default = {
                    'name': 'IMDB', 
                    'verbose': 1, 
                    'top_words': 5000, 
                    'max_review_length': 500, 
                    'nrows': 133, 
                    'start_row': 0, 
                    'metrics': ['accuracy'], 
                    'validation_split': 1/3, 
                    'params': {'dropout': 0.2,
                               'embed_dim': 32,
                               'recurrent_dropout': 0.2, 
                               'epochs': 3, 
                               'num_nodes': 100, 
                               'batch_size': 64, 
                               'activation': 'sigmoid', 
                               'loss_function': 'binary_crossentropy', 
                               'optimizer': 'adam'}},
            **kwargs):
        
        super().__init__(default, **kwargs)

        import classifiers as cla
        import data_wranglers as dat
        
        self.data_object_train = dat.keras_imdb(
                top_words = self.top_words, 
                max_review_length = self.max_review_length, 
                nrows = self.nrows, 
                start_row = self.start_row)
        
        self.data_object_test = dat.keras_imdb(
                top_words = self.top_words, 
                max_review_length = self.max_review_length, 
                nrows = self.nrows, 
                start_row = self.data_object_train.nrows)        

        self.classifier_object = cla.RNN(
                input_dim = int(max(self.data_object_train.data.input.max(),
                                    self.data_object_test.data.input.max())) + 1,
                output_dim = self.data_object_train.data.output.shape[1],
                max_seq_len = max(self.data_object_train.data.input.shape[1],
                                  self.data_object_test.data.input.shape[1]), 
                verbose = self.verbose, 
                validation_split = self.validation_split,
                metrics = self.metrics, 
                params = self.params)        
                
