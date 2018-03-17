#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:36:38 2018

@author: Amine Laghaout
"""
class data_wrangler:
    
    def __init__(self, data = None, **params):
        
        """
        This class wrangles the data into a format which can be readily used
        by machine learning algorithms. In particular it creates a namedtuple
        ``data`` consisting of the input data ``data.input`` (i.e, the 
        features) and the output data ``data.output`` (i.e., the targets).
        
        Parameters
        ----------
        
        data : None, tuple, str
            If ``None``, the data is generated from on the fly. If a tuple, the 
            first element is the input and the second the output. If a string, 
            it represents the path to the data file.
        **params : **kwargs
            Miscellaneous parameters of the data object. These can be the 
            number or rows to load from the data file, the starting row and 
            size of the buffer, etc.
            
        Naming conventions for the parameters ``**params``:
        
        nrows : int
            Number of rows to load
        read_csv : dict
            Dictionary of arguments passed to ``pandas.read_csv`` in the case
            when the data is stored in a CSV file
        feature_names : str, list of str
            Names of the features
        target_names : str, list of str
            Names of the targets
        index : str
            Name of the index, i.e., the field that determines the unique rows

        Returns
        -------
        
        The following are naming conventions for the various attributes.
        
        - ``self.data``: Algorithm-readable data. See ``self.wrangle()``.
        - ``self.data_raw``: Human-readable data. See ``self.wrangle()``.
        - ``self.data_examined``: Extension of the human-readable data after 
          feature engineering. See ``self.examine()``.
        - ``self.feature_profiles``: Statistical profiles of the various 
          features. See ``self.examine()``.
        """
        
        # Prepare the attribute that will contain the data
        from collections import namedtuple
        self.data = namedtuple('data', ['input', 'output'])
       
        # Record all the other parameters as attributes of the data object.
        [self.__setattr__(k, params[k]) for k in params]

        self.wrangle(data)
    
    def wrangle(self, data = None):

        """
        Returns
        -------
        
        self.data : numpy.array
            Numerical data ready to be ingested by the classifier. This is a
            namedtuple that references the input data as ``self.data.input`` 
            and the output data (in the case of supervised learning) as
            ``self.data.output``.
        self.data_raw : utils.Bunch, pandas.DataFrame, etc
            Raw (but clean) data in a human readable form. I.e., unlike 
            ``self.data``, this would not be one hot-encoded.
        """
        
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
            # ``self.read_csv``, if any.
            try:    
                self.data_raw = read_csv(data, **self.read_csv)
            except:
                self.data_raw = read_csv(data)

            # Set the index column                
            try:
                self.data_raw.set_index(self.index, inplace = True)
                
            except: 
                pass
    
    def examine(self):
        
        """
        This function examines the data by performing a statistical analysis on
        the various fields. New features elicited by this examination can be 
        engineered.
        
        Returns
        -------
        
        self.data_examined : pandas.DataFrame
            This is an extension of ``self.data_raw`` which includes new 
            features.
        self.feature_profiles : dict
            Dictionary of statistical profiles for the various features
        """
        
        pass
    
class dummy(data_wrangler):
    
    def wrangle(self, data = None):
        
        self.data_raw = None
        self.data.input = None
        self.data.output = None
        
        print('Wrangling the dummy data...')

class sklearn_dataset(data_wrangler):
    
    """
    This class loads the scikit-learn data set specified as ``data`` to the
    constructor.
    """
    
    def wrangle(self, data = None):
        
        from pandas import DataFrame
        from sklearn import datasets, preprocessing

        # Raw
        dataset = eval('datasets.load_'+data+'()')        
        try:
            dataset.feature_names = dataset.feature_names
        except:
            dataset.feature_names = list(range(dataset.data.shape[1]))
        self.data_raw = DataFrame({x: dataset.data[:, i] for i, x in enumerate(dataset.feature_names)})
        self.data_raw['targets'] = dataset.target
        self.data_raw['target_labels'] =self.data_raw['targets'].apply(lambda x: dataset.target_names[x])
        self.data_raw = self.data_raw.sample(frac = 1)
        
        # Input
        self.data.input = self.data_raw[dataset.feature_names].values
        
        # Output
        label_binarize = preprocessing.LabelBinarizer()
        label_binarize.fit(range(len(dataset.target_names)))
        self.data.output = label_binarize.transform(self.data_raw['targets'])

class strings(data_wrangler):
    
    def wrangle(self, data = None):
        
        super().wrangle(data)
        
        from numpy import array
        from keras.preprocessing import sequence
        
        chardict = dict()
        for i, char in enumerate(self.chars):
            chardict[char] = i + 1
        
        self.data.input = self.data_raw[self.feature_names].map(lambda x: [chardict[y] for y in x])
        self.data.input = sequence.pad_sequences(self.data.input)
        
        try:
            from numpy import zeros, hstack
            extra_padding = zeros((self.data.input.shape[0], 
                                   self.max_seq_len - self.data.input.shape[1]))
            self.data.input = hstack((extra_padding,
                                      self.data.input))
        except:
            pass
        
        self.data.output = array([[x] for x in self.data_raw[self.target_names]])  



class random_walk(data_wrangler):

    """
    Parameters
    ----------

    nrows : int
        Number of walks performed    
    input_dim : int
        Dimension of the input space
    output_dim : int
        Dimension of the output space
    min_seq_len : int
        Minimum number of steps    
    max_seq_len : int
        Maximum number of steps
    max_step : float
        Maximum amplitude of any given step
    
    Returns
    -------
    
    For any given walk, the output can be
    
    - the unique label of the orthant in which the walker ended up (for 
      ``self.output_dim = 1``), or
    - the one-hot encoding of the orthant in which the walker ended up (for
      ``self.output_dim = self.input_dim``), or
    - the position in which the walker ended up (for 
      ``self.output_dim = self.input_dim``), 
    - the distance from the origin at which the walker ended up (for
      ``self.output_dim = 1``).
    
    TODO: 
        So far, only the first scenario is implemented. Implement the three 
        other ones.
    """
    
#    def __init__(self, nrows = 10, input_dim = 2, output_dim = 1, 
#                 min_seq_len = 2, max_seq_len = 20, max_step = 1, verbose = 1):
#        
#            super().__init__(
#                    None, nrows = nrows, input_dim = input_dim, 
#                    output_dim = output_dim, min_seq_len = min_seq_len, 
#                    max_seq_len = max_seq_len, max_step = max_step, 
#                    verbose = verbose)
    
    def wrangle(self, data = None):

        from pandas import DataFrame
        from numpy import append, zeros
        from numpy.random import uniform, choice
                
        self.data_raw = dict()
        self.data.input = zeros((self.nrows, 
                                 self.max_seq_len, 
                                 self.input_dim))
        self.data.output = zeros((self.nrows, self.output_dim))

        # Create variable-length sequences.
        assert 0 < self.min_seq_len < self.max_seq_len
        self.seq_len = choice(range(self.min_seq_len, self.max_seq_len + 1), self.nrows)
        
        from numpy import ones
        self.seq_len = self.max_seq_len*ones(self.nrows, dtype = int)

        # For each walk...
        for row in range(self.nrows):
            
            # Steps and position

            self.data_raw[row] = DataFrame(
                    {str(dim): append([0], uniform(-self.max_step, 
                     self.max_step, self.seq_len[row] - 1)) for dim in range(self.input_dim)})
    
            for dim in self.data_raw[row].columns:
                self.data_raw[row]['position_'+dim] = self.data_raw[row][dim].cumsum()
                
            # Input
        
            self.data.input[row][:self.seq_len[row]] = self.data_raw[row][[str(dim) for dim in range(self.input_dim)]].values
    
            # Output 

            orthant = [self.data_raw[row]['position_'+str(dim)].iloc[-1] > 0 for dim in range(self.input_dim)]
            self.data.output[row] = sum([orthant[dim]*2**dim for dim in range(self.input_dim)])
    
    def plot(self, row = 0):

        """
        Plot the ``row``th random walk.
        
        Parameters
        ----------
        
        row : int
            Index of the random walk
        """
        
        import visualizers as vis
        
        data_raw = self.data_raw[row]
        seq_len = data_raw.shape[0]
        
        if self.input_dim == 2:
            
            vis.plot2D(data_raw.position_0, data_raw.position_1,
                       title = 'Random walk', xlabel = 'Dimension 0',
                       ylabel = 'Dimension 1', legend = False, marker = 'o')
        
        vis.plot2D(list(range(seq_len)), 
                   tuple(data_raw['position_'+str(dim)] for dim in range(self.input_dim)),
                   legend = tuple('Dim. '+str(dim) for dim in range(self.input_dim)),
                   xlabel = 'Step',
                   ylabel = 'Position')
        
class keras_imdb(data_wrangler):

    def wrangle(self, data = None):
    
        from numpy import array
        from numpy.random import seed
        from keras.datasets import imdb
        from keras.preprocessing import sequence
        
        # fix random seed for reproducibility
        seed(7)
        
        # load the dataset but only keep the top n words, zero the rest
        (X_train, y_train), _ = imdb.load_data(num_words = self.top_words)
        
        X_train = X_train[self.start_row:self.start_row + self.nrows]
        y_train = y_train[self.start_row:self.start_row + self.nrows]
        
        # truncate and pad input sequences
        X_train = sequence.pad_sequences(X_train, 
                                         maxlen = self.max_review_length)
        
        # Input
        self.data.input = X_train
        
        # Output
        self.data.output = array([[y] for y in y_train])
        
