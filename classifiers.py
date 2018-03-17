#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:34:34 2018

@author: Amine Laghaout
"""

class classifier:
    
    def __init__(self, **kwargs):
        
        """
        Parameters
        ----------

        - ``self.params``: a namedtuple of model parameters

        Returns
        -------
        
        The following are naming conventions of the main attributes
        
        - ``self.train_curve``: For neural networks, this records the evolution
          of the driving metric over the epochs.
        - ``self.model``: The model (e.g., in scikit-learn, Keras, TensorFlow, 
          etc.)
        - ``self.scores``: The evaluation of the metrics
        """

        # Set the ``kwargs`` as attributes.
        [self.__setattr__(k, kwargs[k]) for k in kwargs]
        
        # Set the parameters as namedtuples.
        from collections import namedtuple
        self.params = namedtuple('params', self.params.keys())(**self.params)
        
        self.build()

    def build(self): 
        
        """
        This function builds the model. E.g., this is a neural network's 
        architecture would get implemented.
        """
        
        pass
    
    def train(self, data_object): 
        
        """
        This function trains the model.
        
        TODO:
        - Add options such as ``shuffle``, ``class_weight``, etc.
        """
        
        self.train_curve = self.model.fit(
                data_object.data.input, 
                data_object.data.output, 
                epochs = self.params.epochs, 
                batch_size = self.params.batch_size,
                validation_split = self.validation_split,
                verbose = self.verbose)
            
    
    def train_report(self, marker = None):
        
        """
        This function provides a report on the trainig process.
        """

        from visualizers import plot2D
        
        epochs = range(1, self.params.epochs + 1)
        
        if self.params.epochs < 20 and marker is None:
            marker = 'o'
            
        plot2D(epochs, 
               (self.train_curve.history['acc'], 
                self.train_curve.history['val_acc']),
               title = 'Model accuracy', xlabel = 'Epoch', ylabel = 'Accuracy',
               legend = ['Train', 'Test'], marker = marker)

        plot2D(epochs, 
               (self.train_curve.history['loss'], 
                self.train_curve.history['val_loss']),
               title = 'Model loss', xlabel = 'Epoch', ylabel = 'Loss',
               legend = ['Train', 'Test'], marker = marker)
                   
    def test(self, data_object): 
        
        """
        This function tests the model.
        """
        
        self.scores = self.model.evaluate(data_object.data.input, 
                                          data_object.data.output,
                                          verbose = self.verbose)
        print('Accuracy: %.2f%%' % (self.scores[1]*100)) 
        
        return self.serve(data_object)
    
    def serve(self, data_object): 
        
        """
        This function serves the model.
        """
        
        return self.model.predict(data_object.data.input)

class dummy(classifier):

    def build(self):
        
        self.model = None
        
        print('Building the dummy classifier...')
        
    def train(self, data_object):
        
        self.train_curve = None
        
        print('Training the dummy classifier...')
        
        pass

    def test(self, data_object):
        
        self.scores = None
        
        print('Testing the dummy classifier...')
        
        pass

    def serve(self, data_object): 
        
        print('Serving the dummy classifier...')
        
        pass

    def train_report(self, marker = None): 
        
        print('Reporting on the training of the dummy classifier...')
        
        pass

class MLP(classifier):
    
    """
    Multi-layer perceptron (aka feedforward neural network)
    """
    
    def build(self):
    
        from keras.optimizers import SGD
        from keras.models import Sequential
        from keras.layers import Dense, Dropout

        self.model = Sequential()

        # Build the neural network layer by layer
        for index, layer in self.params.architecture.iterrows():
            
            # Input layer
            if index == 0:
                
                self.model.add(Dense(int(layer.num_nodes), 
                                     activation = layer.activation, 
                                     input_dim = self.input_dim))
                
                self.model.add(Dropout(layer.dropout))
            
            # Output layer
            elif index == len(self.params.architecture) - 1:

                self.model.add(Dense(self.output_dim, 
                                     activation = layer.activation))
                                
            # Hidden layers
            else:

                self.model.add(Dense(int(layer.num_nodes), 
                                     activation = layer.activation))
                
                self.model.add(Dropout(layer.dropout))
        
        # TODO: Generalize this to any kind of optimizer
        try:
            optimizer = SGD(**self.params.optimizer)
        except:
            from keras.optimizers import Adam
            try:
                optimizer = eval(self.params.optimizer)
            except: 
                optimizer = self.params.optimizer
        
        self.model.compile(loss = self.params.loss_function, 
                           optimizer = optimizer, 
                           metrics = self.metrics)
        
class RNN(classifier):

    """
    Recurrent neural network with embedding
    """
        
    def build(self):

        # LSTM with dropout for sequence classification in the IMDB dataset
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        from keras.layers.embeddings import Embedding
        
        self.model = Sequential()
        
        # Input (embedding) layer
        self.model.add(Embedding(self.input_dim, 
                                 self.params.embed_dim, 
                                 input_length = self.max_seq_len))
        
        # Recurrent layer
        self.model.add(LSTM(self.params.num_nodes, 
                            dropout = self.params.dropout, 
                            recurrent_dropout = self.params.recurrent_dropout))
        
        # Output layer
        self.model.add(Dense(self.output_dim, 
                             activation = self.params.activation))
        
        self.model.compile(loss = self.params.loss_function, 
                           optimizer = self.params.optimizer, 
                           metrics = self.metrics)

class RNN2(classifier):

    """
    Recurrent neural network without embedding
    https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    """
        
    def build(self):

        # LSTM with dropout for sequence classification in the IMDB dataset
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        
        self.model = Sequential()
        
        # Recurrent layer
        self.model.add(LSTM(self.params.num_nodes, 
                            input_shape = (self.max_seq_len, self.input_dim),
                            dropout = self.params.dropout, 
                            recurrent_dropout = self.params.recurrent_dropout))
        
        # Output layer
        self.model.add(Dense(self.output_dim, 
                             activation = self.params.activation))
        
        self.model.compile(loss = self.params.loss_function, 
                           optimizer = self.params.optimizer, 
                           metrics = self.metrics)

class sklearn(classifier):
    
    """
    Support vector classifier
    """

    def build(self):    
    
        if self.algo == 'SVC':
        
            from sklearn.svm import SVC
            self.model = SVC(**self.params)
            
        elif self.algo == 'RandomForestRegressor':
            
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**dict(self.params._asdict()))            
        
    def train(self, data_object):
        
        self.model.fit(data_object.data.input, data_object.data.output)

        from sklearn.feature_selection import SelectKBest, f_regression
        from numpy import log10
        
        predictors = data_object.features
        
        kBest = len(predictors)
        
        selector = SelectKBest(f_regression, k = kBest)   
        selector.fit(data_object.data.input, 
                     data_object.data.output)   
        scores = -log10(selector.pvalues_)
        
        self.feature_relevance = {predictors[i]: p for i, p in enumerate(scores)}
        
    def train_report(self, marker = None):

        pass
    
    def test(self, data_object):

        self.scores = self.model.score(data_object.data.input, 
                                       data_object.data.output)
        prediction = self.serve(data_object)
        prediction = [int(round(x)) for x in prediction.copy()]
        
        print('Accuracy', sum(prediction == data_object.data.output)/len(prediction))
        print('Score:', self.scores) 
        
        return prediction
    
    def serve(self, data_object):
        
        try:
            return self.model.predict(data_object.data.input)
        except:
            return self.model.predict(data_object)