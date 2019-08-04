#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:42:03 2017

@author: Amine Laghaout
"""

from inspect import getargvalues, currentframe
import tensorflow.keras.optimizers as ks_optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers.embeddings import Embedding
from numpy import argmin
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from . import utilities as util


class Estimator:

    def __init__(
            self,
            default_args=None,
            **kwargs):

        util.args_to_attributes(self, default_args, **kwargs)

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

        kwargs = util.parse_args(default_args, kwargs)

        super().__init__(**kwargs)

    def build(self, architecture):

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
            optimizer = ks_optimizers.SGD(**self.optimizer)
        except Exception:

            try:
                optimizer = eval(self.optimizer)
            except Exception:
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

        kwargs = util.parse_args(default_args, kwargs)

        super().__init__(**kwargs)

    def build(self):

        # LSTM with dropout for sequence classification in the IMDB dataset

        self.model = Sequential()

        # With embedding
        try:

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
        except Exception:

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

        args, _, _, values = getargvalues(currentframe())
        values.pop('self')

        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y=None):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

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
