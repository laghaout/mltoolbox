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

    def normalize(self):

        pass

    def reduce(self):
        
        print('Reducing...')

        if self.n_components is not None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_components)
        else:
            pca = None

        return pca

    def select(self):

        pass

    def examine(self):

        pass
    
    def view(self):
        
        print('input:', self.input.shape, 'output:', self.output.shape)
#        print(self.raw.input.head())


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

class Titanic(DataWrangler):

    def __init__(
            self,
            default_args=dict(
                nex=None,
                targets=['Survived'],
                source='./data/titanic/titanic.csv'),
            **kwargs):

        from utilities import parse_args

        kwargs = parse_args(default_args, kwargs)

        super().__init__(**kwargs)

    def human_readable(self):

        from pandas import DataFrame
        from utilities import dict_to_dot, rw_data

        data = rw_data(self.source)

        self.specs = dict_to_dot({
            'input': {feature: dict() for feature
                      in data.columns if feature not in self.targets},
            'output': {target: dict() for target in self.targets}})

        self.raw = dict_to_dot(
            {'input': DataFrame(
                data.iloc[:self.nex], columns=self.specs.input.keys()),
             'output': DataFrame(
                data.iloc[:self.nex][self.targets],
                columns=self.specs.output.keys())})

    def pipe(self):

        from sklearn.pipeline import Pipeline
        
        self.pipeline = [
            ('encode', self.encode())
            ]

        self.pipeline = Pipeline(self.pipeline)


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
    
class TimeSeries(DataWrangler):

    def __init__(
            self,
            default_args=dict(
                name='time series',
                file='./AFT/data/spx_1m_stacked_2015_2016.csv', 
                targets=['SPX_px_last'],
                index='Dt',
                skiprows=None, 
                n_components=None,  # PCA dimension
                kBest=None,         # k best natural features
                scaler=None,
                dropna=True),
            **kwargs):

        from utilities import parse_args

        kwargs = parse_args(default_args, kwargs)

        super().__init__(**kwargs)

    def human_readable(self):

        from utilities import dict_to_dot, rw_data

        data = rw_data(
            self.file, 
            parameters=dict(
                nrows=self.nex,
                skiprows=range(1, self.skiprows))).set_index(self.index)

        if self.dropna:
            data.dropna(inplace=True)

        self.specs = dict_to_dot({
            'input': {feature: dict() for feature in data.columns},
            'output': {target: dict() for target in self.targets}})

        self.raw = dict_to_dot(
            {'input': data[data.columns].iloc[:-1],
             'output': data[self.targets].shift(-1).iloc[:-1]})
    
        self.raw.output.rename(
            columns={target: 'next_'+target for target in self.targets}, 
            inplace=True)
        
        self.targets = self.raw.output.columns.tolist()
    
        if self.nex is None:
            self.nex = self.raw.input.shape[1]
    
    def pipe(self):

        from numpy import ravel, diff, divide
        from sklearn.pipeline import Pipeline
        from utilities import dict_to_dot

        print('Default digits pipeline')
        
        self.A = self.raw.input
        self.output = self.raw.output
        
        # Copy the numerical values from the raw data.
        self.input = self.raw.input.values.copy()
        self.output = ravel(self.raw.output.values.copy())
        
        # For the first four columns, which contain the absolute prices, 
        # compute the rate of change and replace the corresponding part of the
        # machinereadable array.
        self.input[1:, :4] = divide(
            diff(self.input[:, :4], axis=0), self.input[:-1, :4])
        
        # Delete the first row since we cannot compute the rate of change for 
        # it.
        self.input = self.input[1:]
        
        # Compute the rate of change for the output
        self.output = divide(diff(self.output, axis=0), self.output[:-1])
        
        # Delete the first row of the human-readable data since it does not 
        # have any rate of change of its own.
        self.raw.input.drop(self.raw.input.index[0], inplace=True)
        self.raw.output.drop(self.raw.output.index[0], inplace=True)
        
        # Re-adjust the number of rows
        self.nex -= 1

        self.pipeline = dict_to_dot({
            'input': Pipeline([
                ('select', self.select()),
                ('reduce', self.reduce())
                ]),
            'output': None})    

    def select(self, score_func='f_regression'):

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
            
        selection = SelectKBest(score_func, k=self.kBest)

        return selection

    def normalize(self):

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        return scaler
    
    def examine(self):
        
        from pandas import concat, DataFrame
        
        score_funcs = ['f_classif', 'f_regression', 'mutual_info_regression']
        
        feature_scores = DataFrame(
            columns=score_funcs
                + ['linear_relevance', 'linear_uniqueness', 
                   'relevant_uniqueness'],
            index=self.raw.input.columns)

        # Linear correlation matrix
        data = concat((self.raw.input, self.raw.output), axis = 1)
        correlation = data.corr()
        abs_correlation = abs(correlation)

        # Score from scikit-learn's default `SelectKBest` where `K` is the 
        # total number of features
        for score_func in score_funcs:
            selection = self.select(score_func)        
            selection.fit(self.raw.input, self.raw.output)        
            feature_scores[score_func] = selection.scores_
        
        # Relevance based on linear correlation
        feature_scores.linear_relevance = abs_correlation[
            self.raw.output.columns]
    
        # Uniqueness based on linear correlation. Make sure not to sum over
        # the correlation with the target or with the feature itself.
        feature_scores.linear_uniqueness = abs_correlation.apply(
            lambda x: len(x - 2)/(sum(x) - 1 - x[self.raw.output.columns[0]]))
        
        # Relevant uniqueness based on linear correlation
        feature_scores.relevant_uniqueness = feature_scores.apply(
            lambda x: x.linear_relevance * x.linear_uniqueness, axis=1)        
                
        return {'correlation': correlation, 'feature_scores': feature_scores}
        
        
