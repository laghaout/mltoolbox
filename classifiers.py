#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:34:34 2018

@author: Amine Laghaout
"""

def pipeline(classifier_object, examine = False, select = False, train = False, 
             test = False, serve = False, model_summary = False, 
             params = {'marker': 'o'}):
    
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
    
    return_vars : dict
        Dictionary of evaluations
    """
    
    return_vars = dict()
    
    from collections import namedtuple
    params = namedtuple('params', params.keys())(**params)    

    if model_summary:

        print('\n******************** MODEL SUMMARY:')
        classifier_object.model.summary()
    
    if examine is not False:
        
        print('\n******************** EXAMINING:')
        examine.examine() 

    if select is not False:
        
        print('\n******************** SELECTING:')
        classifier_object.select(select)
    
    if train is not False:
        
        print('\n******************** TRAINING:')
        classifier_object.train(train)
        classifier_object.train_report(marker = params.marker)
        return_vars['train'] = {'prediction': classifier_object.test(train)}
        
    if test is not False: 
        
        print('\n******************** TESTING:')        
        return_vars['test'] = {'prediction': classifier_object.test(test)}

    if serve is not False: 
        
        print('\n******************** SERVING:')
        prediction_serve = classifier_object.test(serve)
        return_vars['serve'] = {'prediction': prediction_serve}
        
    return return_vars

class classifier:
    
    def __init__(self, params = None):
        
        """
        This is the generic class for classifiers. The main attributes are 
        
        - ``self.model``: the model (e.g., in scikit-learn, Keras, TensorFlow, 
          etc.)
        - ``self.params``: a namedtuple of model parameters
        
        Parameters
        ----------
        
        params: dict
            Dictionary of the parameters for the model. This includes not only 
            hyperparameters but also other attributes of the model's 
            architechture.
            
        Conventions
        -----------
        
        - ``self.train_curve``: For neural networks, this records the evolution
          of the driving metric over the epochs.
        """
        
        from collections import namedtuple
        self.params = namedtuple('params', params.keys())(**params)
        
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
                validation_split = self.params.validation_split,
                verbose = self.params.verbose)
    
    def train_report(self, marker = 'o'):    
        
        """
        This function provides a report on the trainig process.
        """

        from visualizers import plot2D
        
        epochs = range(1, self.params.epochs + 1)
        
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
                                          verbose = self.params.verbose)
        print("Accuracy: %.2f%%" % (self.scores[1]*100)) 
        
        return self.serve(data_object)
    
    def serve(self, data_object): 
        
        """
        This function serves the model.
        """
        
        return self.model.predict(data_object.data.input)
   
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
                                     input_dim = self.params.input_dim))
                
                self.model.add(Dropout(layer.dropout))
            
            # Output layer
            elif index == len(self.params.architecture) - 1:

                self.model.add(Dense(self.params.output_dim, 
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
            optimizer = eval(self.params.optimizer)
        
        self.model.compile(loss = self.params.loss_function, 
                           optimizer = optimizer, 
                           metrics = self.params.metrics)
        
class RNN(classifier):

    """
    Recurrent neural network
    """

    def build(self):

        # LSTM with dropout for sequence classification in the IMDB dataset
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        from keras.layers.embeddings import Embedding
        
        self.model = Sequential()
        
        # Input (embedding) layer
        self.model.add(Embedding(self.params.input_dim, 
                                 self.params.embed_dim, 
                                 input_length = self.params.max_seq_len))
        
        # Recurrent layer
        self.model.add(LSTM(self.params.num_nodes, 
                            dropout = self.params.dropout, 
                            recurrent_dropout = self.params.recurrent_dropout))
        
        # Output layer
        self.model.add(Dense(self.params.output_dim, 
                             activation = self.params.activation))
        
        self.model.compile(loss = self.params.loss_function, 
                           optimizer = self.params.optimizer, 
                           metrics = self.params.metrics)
