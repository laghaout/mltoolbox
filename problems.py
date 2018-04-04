#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:34:14 2017

@author: Amine Laghaout
"""

class problem:
    
    def __init__(self, default, **kwargs):
        
        # Default arguments        
        [self.__setattr__(k, default[k]) for k in default]
        
        # Arguments specified explicitly. These overwrite the default 
        # arguments.
        [self.__setattr__(k, kwargs[k]) for k in kwargs]
        
        # TODO: Search all the dictionary keys and overwrite if applicable so
        # as to allow for individual parameters nested within the dictionaries
        # to be overwritten.
                     
    def pipeline(self, examine = False, select = False, train = False, 
                 test = False, serve = False, model_summary = False, 
                 params = {'marker': None, 'key_only': False}):
        
        """
        Parameters
        ----------
        
        classifier_object : classifier
            Instance of a classifier object ``classifiers.classifier()``
        examine, select, train, test, serve : data_wrangler, bool
            Instances of a data wrangler object 
            ``data_wranglers.data_wrangler()`` used for data examination, 
            model selection, training, testing, and serving. If these are 
            boolean instead, then a default name for the data object shall be 
            used if ``True``.
        params : dict
            Miscellaneous parameters. E.g., type of marker for the plots, etc.
        
        Returns
        -------
        
        self.results : dict
            Dictionary of evaluations
        """
        
        from collections import namedtuple
        from utilities import get_attributes
        
        self.results = {x: dict() for x in 
                        ('examine', 'select', 'train', 'test', 'serve')}
        params = namedtuple('params', params.keys())(**params)            
        
        print('\n**********************************************************************')    
        print('{:66s} {:4s}'.format(
                '*** '+'Executing the pipeline for '+self.name+'.', '***'))
        print('**********************************************************************')           
        self.results['name'] = self.name
            
        if model_summary:
    
            print('\n******************** MODEL SUMMARY:')
            
            self.classifier_object.model.summary()
        
        if examine is not False:
            
            print('\n******************** EXAMINING:')
            
            if examine is True: examine = self.data_object_train
            
            examine_report = examine.examine()
            self.results['examine'].update(
                    {**examine_report,
                     **{'data': get_attributes(examine, key_only = params.key_only)}})
    
        if select is not False:
            
            print('\n******************** SELECTING:')
            
            if select is True: select = self.data_object_train
                        
            self.classifier_object.select(
                    select) 
            self.results['select'].update(
                    {'data': get_attributes(select, key_only = params.key_only)})
        
        if train is not False:
            
            print('\n******************** TRAINING:')
            
            if train is True: train = self.data_object_train
            
            train_report = self.classifier_object.train(train)
            test_report = self.classifier_object.test(train)
            self.results['train'].update(
                    {**train_report,
                     **{'data': get_attributes(train, key_only = params.key_only)}})
            
            self.classifier_object.train_report(train_report, marker = params.marker)
            self.classifier_object.test_report(test_report)
            
        if test is not False: 
            
            print('\n******************** TESTING:')  
            
            if test is True: test = self.data_object_test
            
            test_report = self.classifier_object.test(test)            
            self.results['test'].update(
                    {**test_report,
                     **{'data': get_attributes(test, key_only = params.key_only)}})
            self.classifier_object.test_report(test_report)
    
        if serve is not False: 
            
            print('\n******************** SERVING:')
            
            if serve is True: serve = self.data_object_serve
            
            self.results['serve'].update(
                    {**self.classifier_object.test(serve),
                     **{'data': get_attributes(serve, key_only = params.key_only)}}) 
            
        return self.results

    def save(self):
        
         pass

    def restore(self):
        
        pass

class dummy(problem): 
    
    """
    Dummy classifier problem. This can be used to trace the pipeline and as
    starter code for new problems.
    """
    
    def __init__(self, 
                 default = {'name': 'dummy problem',
                            'verbose': 0, 
                            'algo': 'some algo', 
                            'input_dim': 5,
                            'output_dim': 4,
                            'nrows_train': 50, 
                            'nrows_test': 50, 
                            'nrows_serve': 50, 
                            'params': {'hyperparam_1': 'ABC',
                                       'hyperparam_2': 60.5}}, 
                 **kwargs):
        
        super().__init__(default, **kwargs)
        
        import classifiers as cla
        import data_wranglers as dat
        
        # Data
        
        data_object_params = {'nrows': self.nrows_train, 
                              'input_dim': self.input_dim, 
                              'output_dim': self.output_dim}
        
        self.data_object_train = dat.dummy(**data_object_params)
        self.data_object_test = dat.dummy(**data_object_params)
        self.data_object_serve = dat.dummy(**data_object_params)
        
        # Model
        
        self.classifier_object = cla.dummy(params = self.params)

class iris(problem): 
    
    """
    Classification of iris types
    """
    
    from pandas import DataFrame
    
    def __init__(
            self,
            default = {
                    'name': 'iris (scikit-learn)',
                    'verbose': 1,
                    'validation_split': 1/3, 
                    'metrics': ['accuracy'],
                    'params_space': {
                            'epochs': [1, 3],
#                            'architecture': {'dropout': [0.1, 0.2, 0.3]}, 
                            'batch_size': [5, 10]
                            }, 
                    'params': {'epochs': 10, 
                               'batch_size': 1, 
                               'loss_function': 'categorical_crossentropy',  
                               'optimizer': 'adam', 
                               'architecture': DataFrame(
                                       {'num_nodes': [16, None], 
                                        'activation': ['sigmoid', 'softmax'], 
                                        'dropout': 0.000123})}},
                 **kwargs):
    
        super().__init__(default, **kwargs)
        
        import classifiers as cla
        import data_wranglers as dat
        
        # Data
        
        self.data_object_train = dat.sklearn_dataset('iris')
        self.data_object_test = dat.sklearn_dataset('iris')
        
        # Model
        
        self.classifier_object = cla.MLP(
                input_dim = self.data_object_train.data.input.shape[1], 
                output_dim = self.data_object_train.data.output.shape[1],  
                validation_split = self.validation_split, 
                verbose = self.verbose,
                metrics = self.metrics, 
                params = self.params,
                params_space = self.params_space)
        
class digits(problem):
    
    """
    Classification of handwritten digits
    """
    
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
        
        # Data
        
        self.data_object_train = dat.sklearn_dataset('digits')
        self.data_object_test = dat.sklearn_dataset('digits')
        
        # Model
        
        self.classifier_object = cla.MLP(
                input_dim = self.data_object_train.data.input.shape[1],
                output_dim = self.data_object_train.data.output.shape[1],
                validation_split = self.validation_split, 
                verbose = self.verbose, 
                metrics = self.metrics, 
                params = self.params)  

class random_walk(problem): 
    
    def __init__(
            self,
            default = {
                    'name': 'random walks',
                    'nrows': 10, 
                    'output_names': 'target', 
                    'input_dim': 2, 
                    'min_seq_len': 2, 
                    'max_seq_len': 20, 
                    'max_step': 1, 
                    'validation_split': 1/3,
                    'metrics': ['accuracy'], 
                    'params': {'num_nodes': 20, 
                               'dropout': 0.2, 
                               'recurrent_dropout': 0.2, 
                               'epochs': 15, 
                               'batch_size': 128, 
                               'activation': 'sigmoid', 
                               'loss_function': 'categorical_crossentropy', 
                               'optimizer': 'adam'},
                    'verbose': 1},
            **kwargs):
        
        super().__init__(default, **kwargs)
        
        import classifiers as cla
        import data_wranglers as dat
        
        data_object_params = {
                'nrows': self.nrows, 
                'output_names': self.output_names, 
                'input_dim': self.input_dim, 
                'min_seq_len': self.min_seq_len,
                'max_seq_len': self.max_seq_len, 
                'max_step': self.max_step,
                'verbose': self.verbose}
        
        self.data_object_train = dat.random_walk(**data_object_params)        
        self.data_object_test = dat.random_walk(**data_object_params)
        self.data_object_serve = dat.random_walk(**data_object_params)
        
        self.classifier_object = cla.RNN(
                input_dim = self.input_dim,
                output_dim = self.data_object_train.data.output.shape[1], 
                max_seq_len = max(self.data_object_train.data.input.shape[1],
                                  self.data_object_test.data.input.shape[1]),
                verbose = self.verbose,
                validation_split = self.validation_split, 
                metrics = self.metrics,
                params = self.params)

class imdb(problem): 

    def __init__(
            self,
            default = {
                    'name': 'IMDB', 
                    'verbose': 1, 
                    'top_words': 5000, 
                    'output_names': 'target', 
                    'max_review_length': 500, 
                    'nrows': 200, 
                    'start_row': 0, 
                    'metrics': ['accuracy'], 
                    'validation_split': 1/3, 
                    'params': {'dropout': 0.2,
                               'embed_dim': 32,
                               'recurrent_dropout': 0.2, 
                               'epochs': 15, 
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
                start_row = self.start_row,
                output_names = self.output_names)
        
        self.data_object_test = dat.keras_imdb(
                top_words = self.top_words, 
                max_review_length = self.max_review_length, 
                nrows = self.nrows, 
                start_row = self.data_object_train.nrows,
                output_names = self.output_names)

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
                    params = self.params_sklearn)            