#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:51:51 2017

@author: Amine Laghaout
"""

import cpuinfo  # python -m pip install -U py-cpuinfo
from datetime import datetime
from json import dump as dump_json
from json import load as load_json
from hyperopt import __version__ as hp_version  # pip install hyperopt
from keras import __version__ as ke_version
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout
from matplotlib import __version__ as plt_version
from numpy import hstack
from numpy import __version__ as np_version
import os
from pandas import DataFrame, MultiIndex, read_csv, read_hdf
from pandas import __version__ as pd_version
from pickle import dump as dump_pickle
from pickle import load as load_pickle
import platform
from sklearn import __version__ as sk_version
from sklearn.preprocessing import LabelBinarizer
from sys import version_info
from tensorflow import __version__ as tf_version
from time import time
from types import SimpleNamespace

from . import visualizers as vis


def generate_windows(start=2008, end=2018):

    timespans = range(start, end)

    ts = []

    for k, start in enumerate(timespans[:-1]):
        ts += [[str(start) + '-01',
                str(start) + '-12',
                str(timespans[k + 1]) + '-01',
                str(timespans[k + 1]) + '-06']]
        ts += [[str(start) + '-06',
                str(timespans[k + 1]) + '-06',
                str(timespans[k + 1]) + '-07',
                str(timespans[k + 1]) + '-12']]

    timespans = DataFrame(ts)
    timespans.columns = MultiIndex.from_product(
        [['train', 'test'],
         ['start', 'end']])

    ts = []

    for k in timespans.index:
        ts += [{
            'train': {
                'start': timespans.iloc[k][('train', 'start')],
                'end': timespans.iloc[k][('train', 'end')]},
            'test': {
                'start': timespans.iloc[k][('test', 'start')],
                'end': timespans.iloc[k][('test', 'end')]}}]

    return ts


def create_model(
        architecture=None,
        input_dim=None,
        output_dim=None,
        optimizer=None,
        loss_function=None,
        metrics=None):

    model = Sequential()

    # Build the neural network layer by layer
    for index, layer in architecture.iterrows():

        # Input layer
        if index == 0:

            model.add(Dense(int(layer.num_nodes),
                            activation=layer.activation,
                            input_dim=input_dim))

            model.add(Dropout(layer.dropout))

        # Output layer
        elif index == len(architecture) - 1:

            model.add(Dense(output_dim,
                            activation=layer.activation))

        # Hidden layers
        else:

            model.add(Dense(int(layer.num_nodes),
                            activation=layer.activation))

            model.add(Dropout(layer.dropout))

    # TODO: Generalize this to any kind of optimizer
    try:
        optimizer = SGD(**optimizer)
    except Exception:

        try:
            optimizer = eval(optimizer)
        except Exception:
            optimizer = optimizer

    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=metrics)

    return model


def version_table(print2screen=True):
    """
    This function returns the version numbers of the various pieces of software
    with which this module was tested.

    Notes
    -----
    In order for Hyperopt 0.1 to work, ``networkx`` had to be downgraded by
    running ``pip install networkx==1.11``. This is due to a bug that arises
    with Hyperopt when version 2.0 of ``networkx`` is installed.

    Also include:
        - conda install plotly

    Parameters
    ----------

    print2screen : bool
        Print the version table to screen (``True``) or return it as a
        dictionary (``False``)?

    Returns
    -------

    version_table : dict
        Dictionary containing the version table
    """

    # TODO: Add Keras

    version_table = {
        'Python': ('3.6.6', '.'.join(str(v) for v in version_info[0:3])),
        'Keras': ('2.1.5', ke_version),
        'TensorFlow.': ('1.6.0', tf_version),
        'NumPy': ('1.14.5', np_version),
        'matplotlib': ('2.2.2', plt_version),
        'sklearn': ('0.20.1', sk_version),
        'PyQt5': ('5.6.2', None),
        'pandas': ('0.23.3', pd_version),
        'Hyperopt': ('0.1', hp_version),
        'OS': ('Linux-4.13.0-17-generic-x86_64-with-debian-stretch-sid',
               platform.platform()),
        'CPU': ('Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz',
                cpuinfo.get_cpu_info()['brand']),
        'CUDA': ('8.0.44', None),
        'GPU': ('NVIDIA GeForce GTX', None)}

    if print2screen:

        # Maximum length of the software names
        pad = max(map(lambda x: len(x), version_table))

        # Print the table.
        print('software'.rjust(pad), ': baseline', sep='')
        print(''.rjust(pad), '  current', sep='')
        for k in sorted(version_table.keys()):
            print(k.rjust(pad), ': ', version_table[k][0], sep='')
            print(''.rjust(pad), '  ', version_table[k][1], sep='')

    return version_table


def encoder(class_names, data=None, binarize_binary=False):
    """
    Parameters
    ----------

    data : list
        List of classes to be binarized
    class_names : list
        List of all the possible classes
    binarize_binary : bool
        If ``True``, apply the binarization to binary classes as well.
        (Otherwise, binary classes will only be assigned a probability scalar.)

    Returns
    -------

    binarized_data : numpy.array
        Binarized numpy array of dimension ``(len(data), len(class_names))``
        corresponding to ``data``
    label_binarizer :
    """

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(class_names)

    if data is not None:

        binarized_data = label_binarizer.transform(data)

        if binarized_data.shape[1] == 1 and binarize_binary:

            binarized_data = hstack((binarized_data, 1 - binarized_data))

        return (binarized_data, label_binarizer)

    else:

        return label_binarizer


def dict_to_dot(dictionary):
    """
    Parameters
    ----------
    dictionary : dict
        Dictionary

    Returns
    -------
    dot : Name space corresponding to the input dictionary
    """

    dot = SimpleNamespace(**dictionary)

    return dot


def args_to_attributes(obj, default_args=None, **kwargs):
    """
    Assign the items of the dictionaries ``default_args`` and ``kwargs`` as
    attributes to the object ``obj``.

    Parameters
    ----------
    obj : object
        Object to which attributes are to be assigned
    default_args : None, str, dict
        Dictionary of default attributes, to be overwritten by ``kwargs``. If
        specified as a string, then it represents the path to the file
        containing dictionary.
    kwargs : dict
        Dictionary of attributes to overwrite the defaults.
    """

    args = parse_args(default_args, kwargs)

    [obj.__setattr__(k, args[k]) for k in args.keys()]

    return obj


def parse_args(default_args, kwargs):
    """
    TODO: Requirements
    - Handle dictionaries only
    - Either manual entries or string with path
    - Default dictionary
    - Allow for fields not in the default dictionary
    - Child class overwrite parent class defaults

    Parameters
    ----------
    default_args : dict, None, str
        Dictionary of default arguments
    kwargs : dict
        Dictionary of new arguments to overwrite the defaults.

    Returns
    -------
    kwargs : dict
        Dictionary of default arguments updated with the new arguments passed
        explicitly
    """

    if default_args is None:
        default_args = dict()
    elif isinstance(default_args, str):
        default_args = rw_data(default_args)

    assert isinstance(default_args, dict)

    kwargs.update(
        {key: default_args[key] for key
         in set(default_args.keys()) - set(kwargs.keys())})

    return kwargs


class Chronometer:
    """
    TODO: The time differences do not make sense. Double-check them.
    """

    def __init__(self):

        self.chrono = DataFrame(
            dict(t=[time()], ts=[datetime.now()], event=['start']))

    def add_event(self, event):

        self.chrono = self.chrono.append(
            dict(t=time(), ts=datetime.now(), event=event),
            ignore_index=True)

    def sort(self):

        self.chrono.sort_values(['t'], inplace=True)
        self.chrono['diff_t'] = self.chrono.t.diff()

    def view(self):

        self.sort()

        print(self.chrono)

        vis.plot_time_series(
            x=self.chrono.event.tolist(),
            y_dict={'diff_t': self.chrono.diff_t}, legend=False,
            ylabel='Time [s]')


def rw_data(path, obj=None, parameters=None):
    """
    Read/write from/to a file.

    See <https://pandas.pydata.org/pandas-docs/stable/io.html>.

    Note that the file must have an extension.

    Parameters
    ----------
    path : str
        Path name of the file. It must start with ``./``.
    obj : generic object
        Object to be read or written
    parameters : dict
        Dictionary of parameters for the IO operation
    """

    extension = path.split('.')[-1].lower()

    # Read
    if obj is None:

        if extension == 'pkl':
            obj = load_pickle(open(path, 'rb'))
        elif extension == 'json':
            obj = load_json(open(path, 'rb'))
        elif extension in {'hdf5', 'h5', 'hdf'}:
            if parameters is None:
                obj = read_hdf(path)
            else:
                obj = read_hdf(path, **parameters)
        elif extension == 'csv':
            if parameters is None:
                obj = read_csv(path)
            else:
                obj = read_csv(path, **parameters)
        else:
            print('WARNING: No file format extension specified')

        return obj

    # Write
    else:

        # Make sure the directory exists

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if extension == 'pkl':
            dump_pickle(obj, open(path, 'wb'))
        elif extension == 'json':
            dump_json(obj, fp=open(path, 'w'))
        elif extension in {'hdf5', 'h5', 'hdf'}:
            obj.to_hdf(path, 'key', mode='w')
        elif extension == 'csv':
            if parameters is None:
                obj.to_csv(path)
            else:
                obj.to_csv(path, **parameters)
        else:
            print('WARNING: No file format extension specified')
