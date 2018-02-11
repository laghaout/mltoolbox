#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:56:20 2017

@author: Amine Laghaout
"""

def load_pickled(path):
    
    from pickle import load
    return load(open(path, 'rb'))
    
def timestamp(as_str = False):
    
    import datetime
    
    now = datetime.datetime.now()
    

    if as_str is not False:
        now = str(now)        
        if as_str == 'YYYY-MM-DD':
            now = now[:10]
    
    return now

def runscript(script, desc = None):
    
    """
    
    """
    
    if type(desc) is str:
        
        print(desc)
    
    if type(script) is str: 
        
        import os
        
        try:
            _ = os.system(script)
        except:
            print('ERROR: <', script, '>', ' failed to run.', sep = '')

def log_to_file(log2file, directory, error_file = 'error_log.txt',
                output_file = 'log.txt'):
    
    """
    This function redirects outputs and errors to a particular file or to the
    console.
    
    Parameters
    ----------
    
    log2file : bool
        If ``True``, save the logs to file.
    directory : str
        Directory where the logs will be saved
    error_file : str
        Name of the error file
    output_file : str
        Name of the output file
    """
    
    if log2file:
        
        import sys
        sys.stderr = open(directory+error_file, 'w')
        sys.stdout = open(directory+output_file, 'w')

        import matplotlib
        matplotlib.use('Agg')
        
    else:
        
        # TODO: Ensure that the errors and stdout are printed to screen.
        pass  

def get_object_settings(self, show = None, no_show = set()):
    
    """
    This function prints all the attributes of the object, or, 
    alternatively all the attributes specified in the set ``show`` but not 
    in the set ``no_show``.
    
    TODO: Implement the selection of the items in ``set{show - no_show}``.
    
    Parameters
    ----------
    
    show : list, set, None
        Attributes that should be shown (``None`` for all)
    no_show : list, set
        Attributes that should not be shown
    """
    
    print('\n***** SETTINGS\n')
    
    for k in sorted(set(self.__dict__.keys()) - set(no_show)):

        attribute = self.__getattribute__(k)
        # TODO: Insert a ``while()`` here to be able to iterate through all
        #       the dictionaries.
        if type(attribute) is dict:
            print('', k, ':', sep = '')
            for k_dict in attribute.keys():
                print('    ', k_dict, ': ', attribute[k_dict], sep = '')
        else:
            print('', k, ': ', attribute, sep = '')
            
    