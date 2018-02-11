#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:52:32 2017

@author: Amine Laghaout
"""

import pickle
from pandas import read_csv, DataFrame, concat

class data_wrangler:
    
    def __init__(self, data_files = None, features = None, bunch_start = 0, 
                 bunch_size = None, verbose = False, name = 'data', 
                 data_raw = None, save_as = None):
        
        """
        Parameters
        ----------
        
        data_files : dict
            Dictionary specifying the data files and the folder where they are 
            placed
        features : list
            List of feature names to be used by the machine learning algorithm
        bunch_start : int
            First row to be read from the raw data
        bunch_size : int
            Number of rows to be read from the raw data starting from 
            ``bunch_start``
        """
            
        if type(features) is str:
            features = [features]
        
        if data_files is None:
            data_files = {'dir': './data/'}

        self.data_files = data_files

        if bunch_size is None:
            
            if data_raw is not None:
            
                bunch_size = len(data_raw)
                
            else: 
                
                from pandas import read_csv
                bunch_size = len(read_csv(self.data_files['dir']+self.data_files['files']).dropna(axis = 'columns'))
        
        self.features = features
        self.bunch_start = bunch_start
        self.bunch_size = bunch_size
        self.verbose = verbose
        self.name = name
        self.data_raw = data_raw
        self.save_as = save_as
        
        # Assert the relationships between nrows, bunch_size, etc
        
        # Try to restore the object if it exists, ...
        try:
        
            self.restore(save_as)
        
        # ... if that fails, re-engineer the object from scratch and save it.
        except:
            
            try:
                
                print('Wrangling...')
                self.wrangle()
                
            except:
                
                print('Direct wrangling failed. Launching one-off wrangling...')
                self.oneoff_wrangle()
                
                print('Re-trying wrangling...')
                self.wrangle()
            
            # Only save data that was not entered manually.
            if self.save_as is not None:
                print('Saving the data [deactivated]...')
                self.save(save_as)
    
    def oneoff_wrangle(self):
    
        pass
        
    def wrangle(self):
        
        """
        This function does one-off data wrangling to prepare the source file
        that will be re-used. It is one-off because one executed, the original
        source files with the messier, more unwieldy data sets, need not be
        processed again.
        """
        
        self.cue()
    
    def cue(self, bunch_start = None, nrows = None, bunch_size = None):

        # The next starting point is ``nrows`` later.
        if bunch_start is True:
            self.bunch_start += self.nrows
            
        # Number of rows to skip from the start or list of rows to skip
        elif bunch_start is not None:            
            self.bunch_start = bunch_start
            
        if nrows is not None:
            self.nrows = nrows
            
        if bunch_size is not None:
            self.bunch_size = bunch_size
    
    def metadata_tag(self, prefix = '', suffix = '', alt_tag = None):
        
        """
        This function returns the tag which helps identify a certain data 
        object. Unless an alternative tag ``alt_tag`` is specified, the default
        tag is `data`.  A ``prefix`` and ``suffix`` can be wrapped around the 
        tag.
        
        Parameters
        ----------
        
        prefix : str
            String to potentially prefix to the tag of the data object
        suffix : str
            String to potentially suffix to the tag of the data object
        alt_tag : str
            Non-default tag
            
        Returns
        -------
        
        tag : str
            tag for the data object
        """
        
        if alt_tag is None:
        
            if prefix != '':
                tag = '_'
            else:
                tag = ''
                
            tag = self.name
            tag += '_'+str(self.bunch_start)+'-' 
            tag += str(self.bunch_start+self.bunch_size)

            if suffix != '':
                tag += '_'
        
        else:
            
            tag = str(alt_tag)
                
        tag = str(prefix)+tag+str(suffix)
        
        return tag

    def save(self, save_as = None, prefix = '', suffix = ''):
        
        """
        This function saves the data wrangler object.
        
        Parameters
        ----------
        
        save_as : str, None
            Name of the file that shall contain the pickled object
        prefix : str
            Prefix to ``save_as``
        suffix : str
            Suffix to ``save_as``
        """
        if type(save_as) is not str:
            save_as = self.data_files['dir']+self.metadata_tag(prefix, suffix, save_as)+'.pickle'
        
        pickle.dump(self, open(save_as, 'wb'))
        
    def restore(self, save_as = None, prefix = '', suffix = ''):

        """
        This function restores the data wrangler object.
        
        Parameters
        ----------
        
        save_as : str, None
            Name of the file that contains the pickled object
        prefix : str
            Prefix to ``save_as``
        suffix : str
            Suffix to ``save_as``        
        """

        if type(save_as) is not str:
            save_as = self.data_files['dir']+self.metadata_tag(prefix, suffix, save_as)+'.pickle'
        
        if self.verbose:
            print('Attempting to restore <', save_as, '>... ', sep = '')
            
        self.__dict__ = pickle.load(open(save_as, 'rb')).__dict__.copy()

    def set_to_digits(self, items):
        
        """
        This function maps each element of a set ``items`` into an integer
        corresponding to its index in the `sorted` set.
        
        Parameters
        ----------
        
        items : set, list, tuple
            Collection of items
        
        Returns
        ------
        
        Dictionary mapping each item to its index in the sorted set of items.
        """
        
        assert type(items) in (list, set, tuple)
        
        return {item: index for index, item in enumerate(sorted(set(items)))}
        
    def get_settings(self, show = None, no_show = set()):
        
        """
        This function prints all the attributes of the object, or, 
        alternatively all the attributes specified in the set ``show`` but not 
        in the set ``no_show``. 
        
        Cf. ``toolbox.get_object_settings()``
        
        Parameters
        ----------
        
        show : list, set, None
            Attributes that should be shown (``None`` for all)
        no_show : list, set
            Attributes that should not be shown        
        """
        
        from toolbox import get_object_settings
        
        get_object_settings(self, show, no_show)
        
    def batch(self, data, batch_start = 0, batch_size = None):
        
        """
        TODO:
            
        - Adapt this function to take in numpy.array data.
        - This function should be integrated into ``evaluate()``.
        
        Parameters
        ----------
        
        data : pandas.DataFrame
            Entire data set of either features or targets
        batch_start : int
            Starting index of the batch from the original set ``data``
        batch_size : int
            Size of the batch
        
        Returns
        -------
        
        A tuple made up of:
        
        batch : pandas.DataFrame, numpy.array
            Batch of either features or targets
        batch_end : int
            Ending index of the batch from the original set ``data``
        """
        
        # The start index and batch size are unspecified: 
        # Use the whole data set.
        if batch_size is None and batch_start is None:
            batch_start = 0
            batch_size = len(data)
            
        # The batch size is unspecified but the start index is specified: 
        # Go to the last item.
        elif batch_size is None and batch_start is not None:
            batch_size = len(data) - batch_start
        
        # The start index is specified but the batch size is unspecified: 
        # Start from 0.
        elif batch_size is not None and batch_start is None:
            batch_start = 0
        
        batch_end = batch_start + batch_size
        
        assert 0 <= batch_start < batch_end <= len(data)
        
        batch = data[batch_start:batch_end]
        
        return (batch, batch_end)
    
    def examine(self, num_lines = 5):

        """
        TODO:

        - Make use of ``sklearn.feature_selection.SelectKBest()`` or any 
          variant thereof. From that, produce a pie chart with the relevance
          of the different features.
        """    
        try:
            print(self.data_raw.head(num_lines))   
        except:
            pass
    
    def split(self, test_proportion = 0.25, n_splits = 3, 
              shuffle = False, random_state = False, stratify = True):

        """
        Split the data into training and testing sets and produce the 
        stratified K-fold  generator for the training set.
        
        Constraints
        -----------
        
        The testing set is extracted from the `end` of the original data set 
        ``data``.
        
        Parameters
        ----------
        
        test_proportion : float, int
            Proportion of the whole data set to be used for testing. It can be 
            a percentage if less than one or an absolute number of data 
            examples if greater or equal to 1.
        n_splits : int
            Number of splits for the K-folding
        shuffle : bool
            Shuffle each stratification of the data before splitting?
        random_state : None, int, or RandomState
            When ``shuffle = true``, pseudo-random number generator state used
            for shuffling. If ``None``, use default NumPy RNG for shuffling.
        
        Returns
        -------
        
        data_train : dict of pandas.DataFrame, dict of numpy.array
            Data used for training (and validation)
        KFolds : sklearn.model_selection._split.StratifiedKFold
            ``KFolds.split(data_train)`` is a generator that specifies the 
            different K-folds for training and validation.
        data_test : dict of pandas.DataFrame, dict of numpy.array
            Data used for testing
        """

        from math import ceil
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # TODO: Generalize this as we're only using dummy data
        from numpy import ones
        data = {'features': 2.6654654*ones(self.nrows),
                'targets': 7.36545498*ones(self.nrows)}
        
        # Determine the number of data examples to extract for the test set.
        if type(test_proportion) is int and 1 <= test_proportion < self.nrows:
            test_start_index = test_proportion
        elif type(test_proportion) is float and 0 <= test_proportion < 1:
            test_start_index = ceil(test_proportion*self.nrows)
        
        test_start_index = self.nrows - test_start_index
        #print('test_start_index:', test_start_index)
        
        data_train = {x: data[x][:test_start_index] for x 
                      in [k for k in data.keys()]}
        
        data_test = {x: data[x][test_start_index:] for x 
                     in [k for k in data.keys()]}
        
        if stratify is True:
            KFolds = StratifiedKFold(shuffle = shuffle, n_splits = n_splits, 
                                     random_state = random_state)
        else:            
            KFolds = KFold(shuffle = shuffle, n_splits = n_splits, 
                           random_state = random_state)
        
        # ``StratifiedKFold()`` requires that the targets be a NumPy array of 
        # a single dimension. If the targets happen to be of more dimensions 
        # than one, collapse their elements by converting them to strings.
        # TODO: Find an alternative to this abominable hackery. Consider, e.g.,
        #       using ``from sklearn.utils.multiclass import unique_labels``.
#        stringed_targets = array([''.join(str(target)) for target 
#                                  in data_train['targets']])
#        KFolds.get_n_splits(data_train['features'], stringed_targets)
        KFolds.get_n_splits()
        
        # Give some advice as to the choice of batch size.        
        if self.verbose == 2:
            
            validation_len = int(len(data_train['features'])/KFolds.n_splits)
            train_len = len(data_train['features'])
            test_len = len(data_test['features'])
            
            print('Choose the batch size judiciously:',
                  'Ensure that it is an integer factor of')
            print('-', validation_len, 'for the validation,')
            print('-', train_len, 'for the training, and')
            print('-', test_len, 'for the testing.')
        
        self.KFolds = KFolds
        
        return KFolds.split(data['features'], data['targets'])

#%% Sepal data
        
class iris(data_wrangler):
    
    def __init__(self, data_files = {'dir': './data/'}, features = None, 
                 bunch_start = 0, bunch_size = None, verbose = True, 
                 name = 'sepal_data', output_size = None, nrows = None):
        
        from pandas import DataFrame
        from sklearn import datasets
        
        iris = datasets.load_iris()

        self.features = iris.feature_names
        self.targets = iris.target_names.tolist()

        self.data_raw = concat([DataFrame(iris.data, columns = self.features), 
                                DataFrame(iris.target, columns = ['Iris'])], axis = 1)
        
        self.bunch_size = len(self.data_raw)
        
        super().__init__(data_files, self.features, bunch_start, bunch_size, 
             verbose, name, data_raw = self.data_raw)
    
    def wrangle(self):        
        
        from numpy import zeros
        
        self.data = {'features': self.data_raw[self.features].values, 
                     'targets': zeros((self.bunch_size, ))}
        
        
        self.data_raw[self.targets].values
    
        pass
