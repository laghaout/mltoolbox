#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:28:39 2020

@author: Amine Laghaout
"""


def args_to_attributes(obj, **kwargs):
    """
    Assign the items of the dictionaries ``default_args`` and ``kwargs`` as
    attributes to the object ``obj``.
    Parameters
    ----------
    obj : object
        Object to which attributes are to be assigned
    kwargs : dict
        Dictionary of attributes to overwrite the defaults.
    """

#    [obj.__setattr__(k, kwargs[k]) for k in kwargs.keys()]

    for k in kwargs.keys():
        print(f'Setting {k}')
        obj.__setattr__(k, kwargs[k])

    return obj
