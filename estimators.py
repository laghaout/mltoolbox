#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:42:03 2018

@author: ala
"""

from sklearn.base import BaseEstimator, ClassifierMixin


class MLP:

    def __init__(self):

        pass

    def build(self):

        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import SGD

        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=64, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy', optimizer=SGD(), 
            metrics=['accuracy'])

        return model


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
