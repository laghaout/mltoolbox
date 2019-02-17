#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 12:51:51 2018

@author: ala
"""

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

    from sklearn.preprocessing import LabelBinarizer

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(class_names)

    if data is not None:

        binarized_data = label_binarizer.transform(data)

        if binarized_data.shape[1] == 1 and binarize_binary:

            from numpy import hstack
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

    from types import SimpleNamespace

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
            from pickle import load
            obj = load(open(path, 'rb'))
        elif extension == 'json':
            from json import load
            obj = load(open(path, 'rb'))
        elif extension in {'hdf5', 'h5', 'hdf'}:
            from pandas import read_hdf
            if parameters is None:
                obj = read_hdf(path)
            else:
                obj = read_hdf(path, **parameters)
        elif extension == 'csv':
            from pandas import read_csv
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
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if extension == 'pkl':
            from pickle import dump
            dump(obj, open(path, 'wb'))
        elif extension == 'json':
            from json import dump
            dump(obj, fp=open(path, 'w'))
        elif extension in {'hdf5', 'h5', 'hdf'}:
            obj.to_hdf(path, 'key', mode='w')
        elif extension == 'csv':
            obj.to_csv(path)
        else:
            print('WARNING: No file format extension specified')
