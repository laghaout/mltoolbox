#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:36:38 2017

@author: Amine Laghaout
"""
class data_wrangler:
    
    def __init__(self, data = None, **params):
        
        """        
        Parameters
        ----------
        
        data : None, tuple, str
            If ``None``, the data is generated from on the fly. If a tuple, the 
            first element is the input and the second the output. If a string, 
            it represents the path to the data file.
        **params : **kwargs
            Miscellaneous parameters of the data object. Some of these 
            parameters can alternatively be generated on the fly based, for 
            example, on the dimensions of the data.
        
        Returns
        -------
        
        All of the parameters ``**params`` are turned into attributes of the 
        data object. The mandatory ones are
        
        self.input_names : list
            List of input names
        self.output_names : list
            List of output names
        self.data_raw : pandas.DataFrame, dict, utils.Bunch
            Human readable data. The input is 
            ``self.data_raw[self.input_names]`` and the output is
            ``self.data_raw[self.output_names]``.
        self.data.input : numpy.array
            Machine-readable input data
        self.data.output : numpy.array
            Machine readable output data
        self.class_names : list
            List of class names
        self.input_dim : int
            Dimensionality of the input. Usually ``self.data.input.shape[1]``.
        self.output_dim : int
            Dimensionality of the output. Usually ``self.data.input.shape[1]``.
        self.nrows : int
            Number of training examples. Usually ``len(self.data_raw)``.
            
        Other, attributes are optional. For example:
        
        self.index : str, list
            Index column for the data
        self.start_row : int
            Starting row for the data
        self.shuffle : bool, float
            Shuffle the rows of the data?
        
        Returns
        -------
        
        self.label_binarize : sklearn.preprocessing.LabelBinarizer
            Mapping between the classes and their one-hot encoding
        """
        
        # Attribute that shall contain the machine-readable data
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
            and, in the case of supervised learning, the output data () as
            ``self.data.output``.
        self.data_raw : pandas.DataFrame, dict, utils.Bunch
            Raw (but cleaned) data in a human-readable form. I.e., unlike 
            ``self.data``, this would not be one hot-encoded.
        """
        
        # If ``data`` is a tuple, ensure that it is made exactly of two 
        # elements, where the first shall be the input and the second the 
        # output.
        if type(data) is tuple:
            
            assert_msg = 'The data has to be a tuple (input, output).'
            assert len(data) == 2, assert_msg
            (self.data.input, self.data.output) = data
        
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
        Returns
        -------
        
        self.data_examined : pandas.DataFrame
            This is an extension of ``self.data_raw`` which includes new 
            features.
        self.input_profiles : dict
            Dictionary of statistical profiles for the various features
        """

        try:
            
            from sklearn.feature_selection import SelectKBest, f_classif
            from numpy import log10
            
            kBest = len(self.input_names)
            
            selector = SelectKBest(f_classif, k = kBest)   
            selector.fit(self.data.input, 
                         self.data_raw[self.output_names])
            scores = -log10(selector.pvalues_)
            
            input_relevance = {self.input_names[i]: p for i, p in enumerate(scores)}
    
            ###
    
            from pandas import DataFrame
            A = input_relevance
            
            B = DataFrame.from_dict(A, orient = 'index').sort_values(
                    0, ascending = False)
            
            import matplotlib.pyplot as plt
            
            fontSize = 16
            top_N = 10
            predictors = B.index.tolist()[:top_N]
            scores = B[0].values[:top_N]
            
            plt.title('Relevance of features')
            plt.barh(range(len(predictors)), scores, align = 'center')
            plt.xlabel('p-values', fontsize = fontSize)
            plt.yticks(range(len(predictors)), predictors, 
                       fontsize = fontSize)
            plt.ylim([-.5,len(predictors)-.5])
            plt.grid()
            plt.show()        
            
            ###
        
        except:
            
            input_relevance = None        
            
        return {'input_relevance': input_relevance,
                'input_profiles': None}
    
class dummy(data_wrangler):
    
    """
    This class generates a dummy data object with random values and generic
    categories.
    """
    
    def wrangle(self, data = None):
        
        from pandas import DataFrame
        from numpy.random import rand, choice
        
        print('Wrangling the dummy data...')
        
        self.input_names = ['feature_'+str(i) for i in range(self.input_dim)]
        self.output_names = 'target'
        self.class_names = ['class_'+str(i) for i in range(self.output_dim)]
        
        # Human-readable data
        
        self.data_raw = DataFrame(
                {**{input_name: rand(self.nrows) for input_name in self.input_names},
                 **{'target': [self.class_names[x] for x in choice(len(self.class_names), self.nrows)]}})
            
        # Machine-readable data
        
        from utilities import binarize        
        self.data.input = self.data_raw[self.input_names].values
        (self.data.output, self.label_binarize) = binarize(
                self.data_raw[self.output_names], self.class_names)         

    def examine(self):
        
        print('Examining the dummy data...')
        
        return super().examine()

class sklearn_dataset(data_wrangler):
    
    """
    This class loads the scikit-learn data set specified as ``data`` to the
    constructor.
    """
    
    def wrangle(self, data = None):
        
        from sklearn import datasets
        from pandas import DataFrame
        from utilities import binarize
        
        dataset = eval('datasets.load_'+data+'()')
        
        # Retrieve the feature names from the dataset if they're readily 
        # available.
        try:
            self.input_names = dataset.feature_names
            
        # If not, just label the features numerically.
        except:
            self.input_names = list(range(dataset.data.shape[1]))
        
        self.class_names = dataset.target_names
        (self.nrows, self.input_dim) = dataset.data.shape
        self.output_dim = len(self.class_names)
        self.output_names = data
        
        # Human-readable data
        
        self.data_raw = DataFrame(
                {**{input_name: dataset.data[:, i] for i, input_name 
                    in enumerate(self.input_names)},
                 **{self.output_names: [dataset.target_names[x] for x in dataset.target]}})
    
        # Shuffle the rows of data        
        self.data_raw = self.data_raw.sample(frac = 1)
        
        # Machine-readable data

        self.data.input = self.data_raw[self.input_names].values
        (self.data.output, self.label_binarize) = binarize(
                self.data_raw[self.output_names], self.class_names)

class strings(data_wrangler):
    
    def wrangle(self, data = None):
        
        super().wrangle(data)
        
        from numpy import array
        from keras.preprocessing import sequence
        
        chardict = dict()
        for i, char in enumerate(self.chars):
            chardict[char] = i + 1
        
        self.data.input = self.data_raw[self.input_names].map(lambda x: [chardict[y] for y in x])
        self.data.input = sequence.pad_sequences(self.data.input)
        
        try:
            from numpy import zeros, hstack
            extra_padding = zeros((self.data.input.shape[0], 
                                   self.max_seq_len - self.data.input.shape[1]))
            self.data.input = hstack((extra_padding,
                                      self.data.input))
        except:
            pass
        
        self.data.output = array([[x] for x in self.data_raw[self.output_names]])      

class random_walk(data_wrangler):

    """
    This random walk is prepared specifically for recurrent neural networks.
    
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
    
    TODO:
    """
    
    def wrangle(self, data = None):

        from pandas import DataFrame
        from numpy import append, zeros
        from sklearn import preprocessing
        from numpy.random import uniform, choice        
                
        self.data_raw = {self.output_names: zeros(self.nrows)}
        self.data.input = zeros((self.nrows, 
                                 self.max_seq_len, 
                                 self.input_dim))
        self.output_dim = 2**self.input_dim
        self.class_names = range(self.output_dim)


        # Create variable-length sequences.
        assert 0 < self.min_seq_len < self.max_seq_len
        self.seq_len = choice(range(self.min_seq_len, self.max_seq_len + 1), self.nrows)
                
        # Fixed-length
        #from numpy import ones
        #self.seq_len = self.max_seq_len*ones(self.nrows, dtype = int) 

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
            self.data_raw[self.output_names][row] = sum([orthant[dim]*2**dim for dim in range(self.input_dim)])

        self.label_binarize = preprocessing.LabelBinarizer()
        self.label_binarize.fit(range(self.output_dim))
        self.data.output = self.label_binarize.transform(self.data_raw[self.output_names])
    
    def examine(self):
        
        self.plot()        
        
        return super().examine()
    
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
        from pandas import DataFrame
        from numpy.random import seed
        from keras.datasets import imdb
        from sklearn import preprocessing
        from keras.preprocessing import sequence
        
        # fix random seed for reproducibility
        seed(7)
        
        self.class_names = range(2)
        
        # load the dataset but only keep the top n words, zero the rest
        (X_train, y_train), _ = imdb.load_data(num_words = self.top_words)
        X_train = X_train[self.start_row:self.start_row + self.nrows]
        y_train = y_train[self.start_row:self.start_row + self.nrows]
        
        # Input: Truncate and pad input sequences
        self.data.input = sequence.pad_sequences(
                X_train, maxlen = self.max_review_length)
        
        # Output
        self.data.output = array([[y] for y in y_train])
        
        self.data_raw = DataFrame(
                {**{self.output_names: self.data.output[:, 0]},
                 **{str(i): self.data.input[:, i] for i in range(self.max_review_length)}})
        
        
            
        self.label_binarize = preprocessing.LabelBinarizer()
        self.label_binarize.fit(self.class_names)
        self.data.output = self.label_binarize.transform(self.data.output)
