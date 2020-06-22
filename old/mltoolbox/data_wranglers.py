#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:36:38 2017

@author: Amine Laghaout
"""

import tensorflow.keras.datasets as keras_dat
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sklearn.feature_selection as skl_feature_selection

from . import utilities as util


class DataWrangler:
    """
    Attributes
    ----------
    input : numpy.array
        Machine-readable input data
    output : numpy.array
        Machine-readable output data
    raw.input : pandas.DataFrame
        Human-readable input data
    raw.output : pandas.DataFrame
        Human-readable output data
    specs.input : dict
        Specifications of the input variables (e.g., type, encode, desc)
    specs.output : dict
        Specifications of the output variables (e.g., type, encode, desc)
    nex : int
        Number of data examples
    index : str
        Name of the index
    """

    def __init__(
            self,
            default_args=dict(nex=1000, encoder=None),
            **kwargs):

        util.args_to_attributes(self, default_args, **kwargs)

        self.verify()
        self.human_readable()
        self.machine_readable()

    def verify(self):
        """
        Check and enforce the consistency of the parameters and attributes of
        the object.
        """

        pass

    def human_readable(self):
        """
        This is where the raw data is loaded and, typically, stored in a human-
        readable fashion.

        Attributes
        ----------
        self.raw.input : DataFrame
        self.raw.output : DataFrame
        """

        data = util.rw_data(self.file)

        self.raw = util.dict_to_dot(
            {'input': data,
             'output': data[self.targets]})

    def machine_readable(self, pipeline=None):
        """
        Attributes
        ----------
        self.input : ndarray
        self.output : ndarray
        """

        # If the pipeline is not specified externally, then use the default
        # pipeline.
        if pipeline is None:
            self.pipe()
        else:
            self.pipeline = pipeline

        if self.pipeline.input is not None:
            self.pipeline.input = self.pipeline.input.fit(
                self.input, self.output)
            self.input = self.pipeline.input.transform(self.input)

        if self.pipeline.output is not None:
            self.pipeline.output = self.pipeline.output.fit(
                self.output)
            self.output = self.pipeline.output.transform(self.output)

    def pipe(self):

        self.input = self.raw.input.values.copy()
        self.output = self.raw.output.values.copy()

        self.pipeline = util.dict_to_dot({'input': None, 'output': None})

    def shuffle(self):
        """ Shuffle or stratify the data. """

        pass

    def encode(self):

        pass

    def impute(self):

        pass

    def normalize(self, scaler=None):

        print('Normalizing...')

        if scaler is True:
            scaler = StandardScaler()
        elif isinstance(scaler, tuple):
            scaler = MinMaxScaler(feature_range=scaler)
        elif scaler is None or scaler is False:
            scaler = None

        return scaler

    def reduce(self):

        print('Reducing...')

        if self.n_components is not None:
            pca = PCA(n_components=self.n_components)
        else:
            pca = None

        return pca

    def select(self, score_func='f_regression'):

        print('Selecting...')

        if self.kBest is None:
            self.kBest = len(self.specs.input)

        assert isinstance(score_func, str)

        if score_func == 'f_regression':
            score_func = skl_feature_selection.f_regression
        elif score_func == 'mutual_info_regression':
            score_func = skl_feature_selection.mutual_info_regression
        elif score_func == 'f_classif':
            score_func = skl_feature_selection.f_classif

        selector = SelectKBest(score_func, k=self.kBest)

        return selector

    def examine(self):

        pass

    def view(self, n_head=3, n_tail=3):
        """
        View the first ``n_head`` and last ``n_tail`` rows of the human-
        readable data.
        """

        if isinstance(n_head, int) or isinstance(n_tail, int):

            print('Input:', self.input.shape)
            if n_head > 0:
                print(self.raw.input.head(n_head))
            if n_tail > 0:
                print(self.raw.input.tail(n_tail))

            print('Output:', self.output.shape)
            if n_head > 0:
                print(self.raw.output.head(n_head))
            if n_tail > 0:
                print(self.raw.output.tail(n_tail))


class FromFile(DataWrangler):

    def __init__(
            self,
            default_args=dict(
                file='https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv',
                targets=['median_house_value']),
            **kwargs):

        kwargs = util.parse_args(default_args, kwargs)

        super().__init__(**kwargs)


class BostonHousing(DataWrangler):

    def __init__(
            self,
            default_args=dict(
                data_set='train',
                test_split=0.2),
            **kwargs):

        kwargs = util.parse_args(default_args, kwargs)

        super().__init__(**kwargs)

    def verify(self):

        assert self.data_set in {'train', 'test', 'serve'}

    @staticmethod
    def oneoff_wrangler(data_set, test_split):

        train, test = keras_dat.boston_housing.load_data(
            test_split=test_split)

        if data_set == 'train':
            return train
        elif data_set == 'test':
            return test

    def human_readable(self):

        (data_in, data_out) = self.oneoff_wrangler(
            self.data_set, self.test_split)

        self.raw = util.dict_to_dot(
            {'input': pd.DataFrame(data_in),
             'output': pd.DataFrame(data_out)})
