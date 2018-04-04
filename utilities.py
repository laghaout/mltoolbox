#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:55:25 2017

@author: Amine Laghaout
"""

def dump_load_pickled(path, obj = None):
    
    """
    Load a pickled file.
    
    Parameters
    ----------
    
    path : str
        Path name of the file
    """
    
    if obj is None:
        
        from pickle import load
        return load(open(path, 'rb'))
    
    else:
        
        import os
        from pickle import dump
                
        os.makedirs(os.path.dirname(path), exist_ok = True)
        dump(obj, open(path, 'wb'))
    

def binarize(data, class_names):
    
    """
    Parameters
    ----------
    
    data : list
        List of classes to be binarized
    class_names : list
        List of all the possible classes
    
    Returns
    -------
    
    binarized_data : numpy.array
        Binarized numpy array of dimension ``(len(data), len(class_names))``
        corresponding to ``data``
    """
    
    from sklearn.preprocessing import LabelBinarizer
    
    label_binarize = LabelBinarizer()
    label_binarize.fit(class_names)
    binarized_data = label_binarize.transform(data)
    
    return (binarized_data, label_binarize)

def get_attributes(self, show = None, no_show = set(), key_only = False,
                   verbose = False, return_value = True):
    
    """
    This function prints all the attributes of the object, or, 
    alternatively all the attributes specified in the set ``show`` but not 
    in the set ``no_show``.
    
    TODO:  
        - Implement the selection of the items in ``set{show - no_show}``.
        - Implement ``key_only``
    
    Parameters
    ----------
    
    show : list, set, None
        Attributes that should be shown (``None`` for all)
    no_show : list, set
        Attributes that should not be shown
    key_only : bool
        Only show the keys and not their values if True
    verbose : bool
        Print dictionary of attributes?
    
    Returns
    -------
    
    attributes : dict
        Dictionary populated with the attributes of ``self``
    """
    
    list_of_attributes = sorted(set(self.__dict__.keys()) - set(no_show))
        
    attributes = {k: None if key_only else self.__getattribute__(k) for k 
                  in list_of_attributes}
    
    if verbose:
    
        print('\n***** ATTRIBUTES\n')
        
        for k in list_of_attributes:
    
            attribute = self.__getattribute__(k)
            # TODO: Insert a ``while()`` here to be able to iterate through all
            #       the dictionaries.
            
            attributes[k] = attribute
            
            if type(attribute) is dict:
                print('', k, ':', sep = '')
                for k_dict in attribute.keys():
                    print('    ', k_dict, ': ', attribute[k_dict], sep = '')
            else:
                print('', k, ': ', attribute, sep = '')

    if return_value:
        return attributes