#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:34:14 2017

@author: Amine Laghaout
"""


class Problem:

    def __init__(self, default_args=None, **kwargs):

        from utilities import args_to_attributes

        args_to_attributes(self, default_args, **kwargs)

        self.verify()
        self.pipe()

    def verify(self):
        """
        Check and enforce the consistency of the parameters and attributes of
        the object.
        """

        pass

    def pipe(self):
        """
        Define the pipeline of estimators.
        """

        pass

    def wrangle(self):

        pass

    def select(self, data_in=None, data_out=None):

        from sklearn.model_selection import GridSearchCV

        if data_in is None:
            data_in = self.data.input
        if data_out is None:
            data_out = self.data.output

        self.search = GridSearchCV(
            self.pipeline, self.params_grid, iid=False, cv=5,
            return_train_score=False)

        self.search.fit(data_in, data_out)

        # Save the best estimator
        self.pipeline = self.search.best_estimator_

        print("Best parameter (CV score=%0.3f):" % self.search.best_score_)
        print(self.search.best_params_)

    def train(self, data_in=None, data_out=None):

        if data_in is None:
            data_in = self.data.train.input
        if data_out is None:
            data_out = self.data.train.output

        self.fitted_pipeline = self.pipeline.fit(data_in, data_out)

    def test(self, data_in=None, data_out=None):

        if data_in is None:
            data_in = self.data.test.input
        if data_out is None:
            data_out = self.data.test.output

        prediction = self.pipeline.predict(data_in)

        if self.data.test.encoder is not None:
            data_out = self.data.test.encoder.inverse_transform(data_out)
        
        self.test_report(prediction, data_out)
        
        return prediction
    
    def test_report(self, prediction, data_out=None):
        
        from sklearn.metrics import accuracy_score
        
        print('Accuracy:', accuracy_score(data_out, prediction))

    def serve(self, data_in=None):

        # TODO: pipeline.predict()

        pass

    def run(self,
            wrangle=True,
            examine=False,
            select=False,
            train=False,
            test=False,
            serve=False):

        if wrangle:
            print('\n**** WRANGLE ****\n')
            self.wrangle()

        if examine:
            print('\n**** EXAMINE ****\n')
            self.examine()

        if select:
            print('\n**** SELECT ****\n')
            self.select()

        if train:
            print('\n**** TRAIN ****\n')
            self.train()

        if test:
            print('\n**** TEST ****\n')
            self.test()

        if serve:
            print('\n**** SERVE ****\n')
            self.serve()


class Digits(Problem):

    from numpy import logspace

    def __init__(
            self,
            default_args=dict(
                name='digits',
                nex=1000,
                algo='SVC',
                params = {
                    'SVC': dict(gamma=1/64),
                    'MLP': dict(epochs=150, batch_size=10, verbose=0)},
                params_grid={
                    'SVC': dict(SVC__gamma=[0.0001, .001, .01, .1]),
                    'MLP': dict(MLP__epochs=[10, 20, 30])}, 
                ),
            **kwargs):

        from utilities import parse_args, dict_to_dot

        kwargs = parse_args(default_args, kwargs)

        kwargs['params_grid'] = kwargs['params_grid'][kwargs['algo']]
        kwargs['params'] = dict_to_dot(kwargs['params'][kwargs['algo']])

        super().__init__(**kwargs)

    def wrangle(self):

        from data_wranglers import Digits

        self.data = Digits(
            nex=self.nex, encoder=True if self.algo == 'MLP' else None)

    def pipe(self):

        from sklearn.pipeline import Pipeline

        if self.algo == 'SVC':

            from sklearn.svm import SVC
    
            self.pipeline = Pipeline(
                [('SVC', SVC(gamma=self.params.gamma))])
    
        elif self.algo == 'MLP':
    
            from keras.wrappers.scikit_learn import KerasClassifier
            from estimators import MLP
    
            MLP_instance = MLP()
    
            model = KerasClassifier(
                build_fn=MLP_instance.build, 
                epochs=self.params.epochs, 
                batch_size=self.params.batch_size, 
                verbose=self.params.verbose)
    
            self.pipeline = Pipeline(
                [('MLP', model)])


class SyntheticClasses(Problem):

    from numpy import logspace

    def __init__(
            self,
            default_args=dict(
                name='synthetic classes',
                n_features=70,
                nex=3000,
                n_redundant=0,
                n_informative=2,
                random_state=1,
                kBest=66,
                n_clusters_per_class=1,
                params_grid=dict(
                    pca__n_components=[5, 20, 30, 40, 45, 50, 55, 64],
                    SVC__gamma=[0.00025*n for n in range(1, 10)]
                    )
                ),
            **kwargs):

        from utilities import parse_args

        kwargs = parse_args(default_args, kwargs)

        super().__init__(**kwargs)

    def wrangle(self):

        from data_wranglers import SyntheticClasses

        self.data = SyntheticClasses(
            n_features=self.n_features,
            nex=self.nex,
            kBest=self.kBest,
            n_redundant=self.n_redundant,
            n_informative=self.n_informative,
            random_state=self.random_state,
            n_clusters_per_class=self.n_clusters_per_class)

    def pipe(self):

        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC

        self.pipeline = Pipeline(
            [('pca', PCA()),
             ('SVC', SVC())])
    
class TimeSeries(Problem):

    from numpy import logspace

    def __init__(
            self,
            default_args=dict(
                name='time series',
                nex={'train': 100000, 'test': 58000},
                skiprows={'train': 255, 'test': 100000 + 255},
                algo='RFR',
                kBest=None,
                n_components=None,
                params = {
                    'RFR': dict(n_estimators=10),
                    'RNN': dict(epochs=150, batch_size=10, verbose=0)},
                params_grid={
                    'RFR': dict(n_estimators=[5, 10, 15, 20]),
                    'RNN': dict(MLP__epochs=[10, 20, 30])}, 
                ),
            **kwargs):

        from utilities import parse_args, dict_to_dot

        kwargs = parse_args(default_args, kwargs)

        kwargs['params_grid'] = kwargs['params_grid'][kwargs['algo']]
        kwargs['params'] = dict_to_dot(kwargs['params'][kwargs['algo']])
        
        for param in ('nex', 'skiprows'):
            kwargs[param] = dict_to_dot(kwargs[param])

        super().__init__(**kwargs)
        
    def verify(self):
        
        assert self.skiprows.test == self.skiprows.train + self.nex.train

    def wrangle(self):

        from data_wranglers import TimeSeries
        from utilities import dict_to_dot
    
        self.data = dict_to_dot({
            'train': TimeSeries(
                n_components=self.n_components, kBest=self.kBest, 
                nex=self.nex.train, skiprows=self.skiprows.train),
            'test': TimeSeries(
                n_components=self.n_components, kBest=self.kBest, 
                nex=self.nex.test, skiprows=self.skiprows.test)})

    def pipe(self):

        from sklearn.pipeline import Pipeline

        if self.algo == 'RFR':

            from sklearn.ensemble import RandomForestRegressor as RFR
    
            self.pipeline = Pipeline(
                [('RFR', RFR())])
    
        elif self.algo == 'MLP':
    
            from keras.wrappers.scikit_learn import KerasClassifier
            from estimators import MLP
    
            MLP_instance = MLP()
    
            model = KerasClassifier(
                build_fn=MLP_instance.build, 
                epochs=self.params.epochs, 
                batch_size=self.params.batch_size, 
                verbose=self.params.verbose)
    
            self.pipeline = Pipeline(
                [('MLP', model)])
            
    def test_report(self, prediction, data_out):
        
        from sklearn.metrics import mean_squared_error
        from visualizers import plotTimeSeries
        
        mse = mean_squared_error(data_out, prediction)
        print('MSE:', mse)

        plotTimeSeries(
            x=self.data.test.raw.output.index, 
            y_dict={'prediction': prediction.cumsum(), 
                    'actual': data_out.cumsum()},
            linewidth=3, save_as='./cumsum_ROC_test.pdf',
            title='MSE '+str(mse))
