#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:42:03 2018

@author: Amine Laghaout
"""

from sklearn.base import BaseEstimator, ClassifierMixin

class Estimator:
    
    def __init__(
            self,
            default_args=None,
            **kwargs):

        from utilities import args_to_attributes

        args_to_attributes(self, default_args, **kwargs)

        self.verify()    

    def verify(self):
        
        pass

    def build(self):
        
        pass
    
class MLP(Estimator):

    def __init__(
            self,
            default_args=dict(
                name='mutli-layer perceptron',
                ),
            **kwargs):

        from utilities import parse_args

        kwargs = parse_args(default_args, kwargs)

        super().__init__(**kwargs)

    def build(self, architecture):
        
        from keras.optimizers import SGD
        from keras.models import Sequential
        from keras.layers import Dense, Dropout

        self.architecture = architecture

        self.model = Sequential()

        # Build the neural network layer by layer
        for index, layer in self.architecture.iterrows():

            # Input layer
            if index == 0:

                self.model.add(Dense(int(layer.num_nodes),
                                     activation=layer.activation,
                                     input_dim=self.input_dim))

                self.model.add(Dropout(layer.dropout))

            # Output layer
            elif index == len(self.architecture) - 1:

                self.model.add(Dense(self.output_dim,
                                     activation=layer.activation))

            # Hidden layers
            else:

                self.model.add(Dense(int(layer.num_nodes),
                                     activation=layer.activation))

                self.model.add(Dropout(layer.dropout))

        # TODO: Generalize this to any kind of optimizer
        try:
            optimizer = SGD(**self.optimizer)
        except BaseException:
            from keras.optimizers import Adam
            try:
                optimizer = eval(self.optimizer)
            except BaseException:
                optimizer = self.optimizer

        self.model.compile(loss=self.loss_function,
                           optimizer=optimizer,
                           metrics=self.metrics)

        return self.model
    
    
class RNN(Estimator):

    def __init__(
            self,
            default_args=dict(
                name='recurrent neural network',
                ),
            **kwargs):

        from utilities import parse_args

        kwargs = parse_args(default_args, kwargs)

        super().__init__(**kwargs)
        
    def build(self):

        # LSTM with dropout for sequence classification in the IMDB dataset
        from keras.models import Sequential
        from keras.layers import Dense, LSTM

        self.model = Sequential()

        # With embedding
        try:

            from keras.layers.embeddings import Embedding

            # Input (embedding) layer
            self.model.add(Embedding(self.input_dim,
                                     self.embed_dim,
                                     input_length=self.max_seq_len))

            # Recurrent layer
            self.model.add(
                LSTM(
                    self.num_nodes,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout))

        # Without embedding
        # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        except BaseException:

            # Recurrent layer
            self.model.add(
                LSTM(
                    self.num_nodes,
                    input_shape=(
                        self.max_seq_len,
                        self.input_dim),
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout))

        # Output layer
        self.model.add(Dense(self.output_dim,
                             activation=self.activation))

        self.model.compile(loss=self.loss_function,
                           optimizer=self.optimizer,
                           metrics=self.metrics)
        
        return self.model


class TemplateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, arg_1=1000, arg_2=5):

        from inspect import getargvalues, currentframe

        args, _, _, values = getargvalues(currentframe())
        values.pop('self')

        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y=None):

        from sklearn.utils.multiclass import unique_labels
        from sklearn.utils.validation import check_X_y

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        from numpy import argmin
        from sklearn.metrics import euclidean_distances
        from sklearn.utils.validation import check_array, check_is_fitted

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

#    def score(self):
#
#        # TODO
#
#        pass
