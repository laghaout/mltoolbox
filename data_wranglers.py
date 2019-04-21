#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:36:38 2017

@author: Amine Laghaout
"""


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

        from utilities import args_to_attributes

        args_to_attributes(self, default_args, **kwargs)

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

        pass

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
        
        from utilities import dict_to_dot
        
        self.input = self.raw.input.values.copy()
        self.output = self.raw.output.values.copy()

        self.pipeline = dict_to_dot({'input': None, 'output': None})

    def shuffle(self):
        """
        Shuffle or stratify the data.
        """

        pass

    def encode(self):

        pass

    def impute(self):

        pass

    def normalize(self, scaler=None):

        print('Normalizing...')
        
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
               
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
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_components)
        else:
            pca = None

        return pca

    def select(self, score_func='f_regression'):

        print('Selecting...')
        
        from sklearn.feature_selection import SelectKBest

        if self.kBest is None:
            self.kBest = len(self.specs.input)
        
        assert isinstance(score_func, str)
        
        if score_func == 'f_regression':
            from sklearn.feature_selection import f_regression as score_func           
        elif score_func == 'mutual_info_regression':
            from sklearn.feature_selection import mutual_info_regression as score_func
        elif score_func == 'f_classif':
            from sklearn.feature_selection import f_classif as score_func
            
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
                
class Factorizable(DataWrangler):
    
    def __init__(
            self,
            default_args=dict(
                factor=7,
                min_int=-1000,
                max_int=1000,
                nex=5000,
                ncols=2),
            **kwargs):
        
        from utilities import parse_args

        kwargs = parse_args(default_args, kwargs)

        super().__init__(**kwargs)
    
    def human_readable(self):
        
        from numpy.random import randint
        from pandas import DataFrame
        from utilities import dict_to_dot

        self.raw = dict_to_dot({'input': None, 'output': None})
        
        self.raw.input = DataFrame(
            {x: randint(self.min_int, self.max_int, self.nex) for x 
             in range(self.ncols)})
        self.raw.input['sum'] = self.raw.input.apply(
            lambda x: sum([x[k] for k in range(self.ncols)]), axis=1)
        self.raw.output = self.raw.input.apply(
            lambda x: x['sum'] % self.factor == 0, axis=1)
        del self.raw.input['sum']
        
class Digits(DataWrangler):

    def __init__(
            self,
            default_args=dict(
                nex=800,
                encoder=True,
                n_components=17),
            **kwargs):       

        from utilities import parse_args

        kwargs = parse_args(default_args, kwargs)

        super().__init__(**kwargs)

    def human_readable(self):

        from pandas import DataFrame
        from utilities import dict_to_dot
        from sklearn import datasets

        digits = datasets.load_digits()
    
        self.specs = dict_to_dot({
            'input': {'pixel_'+str(n): dict() for n
                      in range(1, digits.data.shape[1] + 1)},
            'output': {'target_1': dict()}})
        
        self.raw = dict_to_dot(
            {'input': DataFrame(
                digits.data[:self.nex], columns=self.specs.input.keys()),
             'output': DataFrame(
                digits.target[:self.nex], columns=self.specs.output.keys())})
    
        self.shuffle()
    
    def shuffle(self):
        
        print('Shuffling...')
        
        self.raw.input = self.raw.input.sample(frac=1)
        self.raw.output = self.raw.output.loc[self.raw.input.index]
    
    def encode(self):
        
        print('Encoding...', self.encoder)
                   
        if self.encoder is True:
            from utilities import encoder
            (self.output, self.encoder) = encoder(range(10), self.output)
        elif self.encoder is False or self.encoder is None:
            pass
        else:
            self.output = self.encoder.transform(self.output)
               
    def pipe(self):

        from sklearn.pipeline import Pipeline
        from utilities import dict_to_dot

        print('Default digits pipeline')
        
        self.input = self.raw.input.values.copy()
        self.output = self.raw.output.values.copy()

        self.pipeline = dict_to_dot({
            'input': Pipeline([
                ('reduce', self.reduce()),
                ]),
            'output': Pipeline([
                ('encode', self.encode())])})

class SyntheticClasses(DataWrangler):

    def __init__(
            self,
            default_args=dict(
                name='synthetic classes',
                scaler=None,
                n_features=20,
                n_redundant=0,
                n_informative=2,
                n_classes=2,
                random_state=1,
                kBest=None,
                n_clusters_per_class=1),
            **kwargs):

        from utilities import parse_args, dict_to_dot

        kwargs = parse_args(default_args, kwargs)

        kwargs['specs'] = dict_to_dot({
            'input': {'feature_'+str(n): dict() for n
                      in range(1, kwargs['n_features'] + 1)},
            'output': {'target_1': dict()}})

        super().__init__(**kwargs)

    def verify(self):

        assert self.n_features == len(self.specs.input)

    def human_readable(self):

        from pandas import DataFrame
        from sklearn.datasets import make_classification
        from utilities import dict_to_dot

        data = make_classification(
            n_samples=self.nex,
            n_features=len(self.specs.input),
            n_redundant=self.n_redundant,
            n_informative=self.n_informative,
            random_state=self.random_state,
            n_classes=self.n_classes,
            n_clusters_per_class=self.n_clusters_per_class)

        self.raw = dict_to_dot(
            {'input': DataFrame(
                data[0], columns=self.specs.input.keys()),
             'output': DataFrame(
                data[1], columns=self.specs.output.keys())})

    def pipe(self):

        from sklearn.pipeline import Pipeline

        self.pipeline = [
            ('select', self.select())
            ]

        self.pipeline = Pipeline(self.pipeline)

    def select(self):

        from sklearn.feature_selection import SelectKBest

        if self.kBest is None:
            self.kBest = self.n_features

        selection = SelectKBest(k=self.kBest)

        return selection

    def normalize(self):

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        return scaler

    def reduce(self):

        from sklearn.decomposition import PCA, KernelPCA
        from sklearn.pipeline import FeatureUnion

        union = [
            ('pca', PCA()),
            ('kpca', KernelPCA(kernel='rbf'))
            ]

        union = FeatureUnion(union)

        return union    
    
