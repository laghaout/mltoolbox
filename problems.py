#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:34:14 2017

@author: Amine Laghaout
"""


class Problem:

    def __init__(
            self, 
            default_args=None, **kwargs):

        from utilities import args_to_attributes, Chronometer

        args_to_attributes(self, default_args, **kwargs)

        self.chrono = Chronometer()

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

    def select(self, data=None):

        from sklearn.model_selection import GridSearchCV

        if data is None:
            data = self.data.train

        self.search = GridSearchCV(
            self.pipeline, self.params_grid, iid=False, cv=3,
            return_train_score=False)

        self.search.fit(data.input, data.output)

        # Save the best estimator
        self.pipeline = self.search.best_estimator_

        print("Best parameter (CV score=%0.3f):" % self.search.best_score_)
        print(self.search.best_params_)
        
    def train(self, data=None):
        
        if data is None:
            data = self.data.train
            
        self.pipeline.fit(data.input, data.output)
        
        self.train_report()
        
        self.test(data)
        
    def train_report(self):
        
        from visualizers import plotTimeSeries
        
        if self.algo in ['MLP', 'RNN']:        
            history = self.pipeline.named_steps[self.algo].model.history
        
            plotTimeSeries(
                x=history.epoch, 
                y_dict={x: history.history[x] for x in history.history.keys()},
                xlabel='epoch')
        
    def test(self, data=None):

        if data is None:
            data = self.data.test
            
        prediction = self.serve(data)
        
        return self.test_report(data, prediction)
    
    def test_report(self, actual_data, predicted_data):
        
        # Check whether the actual data is a ``data_wrangler`` format object. 
        # If not, then use it as is.
        try:
            actual_data = actual_data.raw.output
        except:
            pass
               
        # Classification
        try:
            from sklearn.metrics import accuracy_score
            
            accuracy = accuracy_score(actual_data, predicted_data)
            print('Accuracy:', accuracy)
            
            report = dict(accuracy=accuracy)
            
        # Regression
        except:
            from sklearn.metrics import mean_squared_error
            
            mse = mean_squared_error(actual_data, predicted_data)
            print('MSE:', mse)
            
            report = dict(mse=mse)
        
        return report

    def serve(self, data):

        if data is None:
            data = self.data.serve        

        prediction = self.pipeline.predict(data.input)

        if data.pipeline.output is not None:
            if 'normalize' in data.pipeline.output.named_steps.keys():
                if data.pipeline.output.named_steps['normalize'] is not None:
                    prediction = data.pipeline.output.named_steps['normalize'].inverse_transform(prediction)
        
        return prediction         

    def run(self,
            wrangle=True,
            examine=False,
            select=False,
            train=False,
            test=False,
            serve=False):

        print('********', self.name, '**********')

        self.report = dict()

        if wrangle:
            print('\n**** WRANGLE ****\n')
            self.chrono.add_event('start wrangle')
            self.report['wrangle'] = self.wrangle()
            self.chrono.add_event('end wrangle')

        if examine:
            print('\n**** EXAMINE ****\n')
            self.chrono.add_event('start examine')
            self.report['examine'] = self.examine()
            self.chrono.add_event('end examine')

        if select:
            print('\n**** SELECT ****\n')
            self.chrono.add_event('start select')
            self.report['select'] = self.select()
            self.chrono.add_event('end select')

        if train:
            print('\n**** TRAIN ****\n')
            self.chrono.add_event('start train')
            self.report['train'] = self.train()
            self.chrono.add_event('end train')

        if test:
            print('\n**** TEST ****\n')
            self.chrono.add_event('start test')
            self.report['test'] = self.test()
            self.chrono.add_event('end test')

        if serve:
            print('\n**** SERVE ****\n')
            self.chrono.add_event('start serve')
            self.report['serve'] = self.serve()
            self.chrono.add_event('end serve')
            
#        self.chrono.view()


        
class Digits(Problem):

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
