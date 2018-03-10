#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:36:38 2018

@author: Amine Laghaout
"""

class data_wrangler:
    
    def __init__(self, data = None, params = None):
        
        """
        This class wrangles the data into a format which can be readily used
        by machine learning algorithms. In particular it creates a namedtuple
        ``data`` consisting of the input data ``data.input`` (i.e, the 
        features) and the output data ``data.output`` (i.e., the targets).
        
        Parameters
        ----------
        
        data : None, tuple, str
            If ``None``, the data is generated from some package on the fly. If
            a tuple, the first element is the input and the second the output. 
            If a string, it represents the path to the data file.
        params : 
            Miscellaneous parameters of the data object. These can be the 
            number or rows to load from the data file, the starting row and 
            size of the buffer, etc.
        
        Conventions
        -----------
        
        Some additional properties of the object can be added according to the
        following convention:
        
        self.data_raw : utils.Bunch, pandas.DataFrame, etc
            Raw (but clean) data in a human readable form. I.e., unlike 
            ``self.data``, this would not be one hot-encoded.
        """
        from collections import namedtuple

        self.data = namedtuple('data', ['input', 'output'])
        
        # Turn the dictionary of parameters into a namedtuple.
        try:
            from collections import namedtuple
            self.params = namedtuple('params', params.keys())(**params)
        except:
            self.params = params

        self.wrangle(data)
    
    def wrangle(self, data = None):

        # If ``data`` is a tuple, ensure that it is made to exactly two 
        # elements, where the first shall be the input and the second the 
        # output.
        if type(data) is tuple:
            
            assert_msg = 'The data has to be a tuple (input, output).'
            assert len(data) == 2, assert_msg
            self.data.input, self.data.output = data
        
        # If ``data`` is a string, then it represent the path name of the data
        # file.
        elif type(data) is str:
            
            from pandas import read_csv
            
            # Read the CSV file according to the arguments passed as
            # ``self.params.read_csv``, if any.
            try:    
                self.data_raw = read_csv(data, **self.params.read_csv)
            except:
                self.data_raw = read_csv(data)

            # Set the index column                
            try:
                self.data_raw.set_index(self.params.index, inplace = True)
            except: 
                pass
    
    def examine(self):
        
        """
        This function examines the data by performing a statistical analysis on
        the various fields.
        """
        
        pass

class sklearn_dataset(data_wrangler):
    
    """
    This class loads the scikit-learn data set specified as ``data`` to the
    constructor.
    """
    
    def wrangle(self, data = None):
        
        from sklearn import datasets, preprocessing

        # Raw
        self.data_raw = eval('datasets.load_'+data+'()')
        
        # Input
        self.data.input = self.data_raw.data
        
        # Output
        label_binarize = preprocessing.LabelBinarizer()
        label_binarize.fit(range(len(self.data_raw.target_names)))
        self.data.output = label_binarize.transform(self.data_raw.target)


class strings(data_wrangler):
    
    def wrangle(self, data = None):
        
        super().wrangle(data)
        
        from numpy import array
        from keras.preprocessing import sequence
        
        chardict = dict()
        for i, char in enumerate(self.params.chars):
            chardict[char] = i + 1
        
        self.data.input = self.data_raw[self.params.feature_name].map(lambda x: [chardict[y] for y in x])
        self.data.input = sequence.pad_sequences(self.data.input)
        
        try:
            from numpy import zeros, hstack
            extra_padding = zeros((self.data.input.shape[0], 
                                   self.params.max_seq_len - self.data.input.shape[1]))
            self.data.input = hstack((extra_padding,
                                      self.data.input))
        except:
            pass
        
        self.data.output = array([[x] for x in self.data_raw[self.params.target_name]])  
        
class keras_imdb(data_wrangler):

    def wrangle(self, data = None):
    
        import numpy
        from keras.datasets import imdb
        from keras.preprocessing import sequence
        
        # fix random seed for reproducibility
        numpy.random.seed(7)
        
        # load the dataset but only keep the top n words, zero the rest
        (X_train, y_train), _ = imdb.load_data(
                num_words = self.params.top_words)
        
        # truncate and pad input sequences
        X_train = sequence.pad_sequences(
                X_train, maxlen = self.params.max_review_length)
        
        # Input
        self.data.input = X_train
        
        # Output
        self.data.output = y_train
        
