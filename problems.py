#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:34:14 2017

@author: Amine Laghaout
"""


class Problem:
    """
    This class represents a generic machine learning problem as it lines up the 
    various stages, namely 
    
    - data wrangling with ``wrangle()``, 
    - data examination with ``examine()``, 
    - model selection with ``select()``, 
    - model training with ``train()``,
    - model testing with ``test()``, and
    - model serving with ``serve()``
    
    into a single "pipeline" which can be invoked by ``run()``.
    
    The arguments to the problem object as well as to all its constiuent 
    functions above are passed either as a dictionary or as a string specifying
    the path to the dictionary. These arguments are processed in 
    ``__init__()`` and consistency-checked by ``verify()``.
    
    The machine learning model itself is assembled in ``pipe()``.
    """

    def __init__(
            self, 
            default_args=None, **kwargs):

        from utilities import args_to_attributes, Chronometer

        # Supersede the default arguments ``default_args`` with the arguments
        # passed as ``**kwargs``. All these  arguments will then be attributed 
        # to the object.
        args_to_attributes(self, default_args, **kwargs)

        # Launch the chronometer for the problem.
        self.chrono = Chronometer()

        # Verify the consistency and integrity of the arguments.
        self.verify()
        
        # Assemble the model pipeline (typically with a scikit-learn pipeline).
        self.pipe()

    def verify(self):
        """
        Verify the consistency and integrity of the arguments attributed to the
        object.
        """

        pass

    def pipe(self):
        """
        Define the pipeline of estimators. This is typically where a scikit-
        learn pipeline is defined and stored as ``self.pipeline``.
        """

        pass

    def wrangle(self):
        """
        Wrangle the data. This is where the data objects for training, testing,
        and serving could be defined and stored as, say, 
        ``self.data.{train, test, serve}``.
        """

        pass
    
    def examine(self):
        """
        Examine the data. E.g., this is where exploratory statistical analysis 
        of the data is performed.
        """

        pass    

    def select(self, data=None, update_with_best=True):
        """
        Select the best model. This is where the hyperparameter selection is
        is performed.
        
        Parameters
        ----------
        data : None, ``data_wrangler.DataWrangler``, numpy.array
            Data to be selected ib. If None, the training data specified at the 
            wrangling stage is used.    
        update_with_best : bool
            If True, replace the 
        """

        from sklearn.model_selection import GridSearchCV

        # If the data is not specified, then use the training data already 
        # specified in the wrangling stage.
        if data is None:
            data = self.data.train

        # Create the hyperparameter search object and fit it to the data.
        self.search = GridSearchCV(
            self.pipeline, self.params_grid, iid=False, cv=3,
            return_train_score=False)
        self.search.fit(data.input, data.output)

        # Save the best estimator.
        if update_with_best:
            self.pipeline = self.search.best_estimator_

        print('Best parameter (CV score=%0.3f):' % self.search.best_score_)
        print(self.search.best_params_)
        
    def train(self, data=None):
        """
        Train the model.
        
        Parameters
        ----------
        data : None, ``data_wrangler.DataWrangler``, numpy.array
            Data to be trained on. If None, the training data specified at the 
            wrangling stage is used.        
        """
        
        # If the data is not specified, then use the training data already 
        # specified in the wrangling stage.
        if data is None:
            data = self.data.train
        
        self.pipeline.fit(data.input, data.output)
        
        # Report on the training.
        self.train_report()
        
        # Test the model on the training set.
        self.test(data)
        
    def train_report(self):
        """
        Report on the training.
        """
        
        from visualizers import plotTimeSeries
        
        # If the algorithm is a neural network in Keras, retreive the training
        # history.
        if self.algo in ['MLP', 'RNN']:        
            
            history = self.pipeline.named_steps[self.algo].model.history
        
            plotTimeSeries(
                x=history.epoch, 
                y_dict={x: history.history[x] for x in history.history.keys()},
                xlabel='epoch')
        
    def test(self, data=None):
        """
        Test the model.

        Parameters
        ----------
        data : None, ``data_wrangler.DataWrangler``, numpy.array
            Data to be tested on. If None, the testing data specified at the 
            wrangling stage is used.
        
        Returns
        -------
        report : dict
            Test report.
        """

        # If the data is not specified, then use the testing data already 
        # specified in the wrangling stage.
        if data is None:
            data = self.data.test
        
        # Generate the prediction.
        prediction = self.serve(data)
        
        print('prediction:', prediction[:5], prediction.shape)
        
        # Compare the prediction with the actual test data.
        report = self.test_report(data, prediction)
        
        return report
    
    def test_report(self, actual_data, predicted_data):
        """
        Report on the testing.
        
        TODO: Replace the try~except blocks with something more elegant.

        Parameters
        ----------
        actual_data : ``data_wrangler.DataWrangler``, numpy.array
            Data to be tested on
        predicted_data : numpy.array
            Predicted data
        
        Returns
        -------
        report : dict
            Test report.
        """
        
        # Check whether the actual data is a ``data_wrangler.DataWrangler`` 
        # object. If it is, extract only the raw output. Otherwise, use it as
        # is.
        try:
            actual_data = actual_data.raw.output
        except BaseException:
            pass

        # Check whether the problem is a classfication, in which case the test
        # data is assessed by the confusion matrix.
        try:
            # TODO: Replace the accuracy with the confusion matrix.
            from sklearn.metrics import accuracy_score
            
            accuracy = accuracy_score(actual_data, predicted_data)
            print('Accuracy:', accuracy)
            
            report = dict(accuracy=accuracy)
            
        # If the problem is not a classification, then assume it is a
        # regression and evaluate it with the mean squared error.
        except:
            from sklearn.metrics import mean_squared_error

            print('actual_data:', actual_data[:5], actual_data.shape)
            print('predicted_data:', predicted_data[:5], predicted_data.shape)
            
            mse = mean_squared_error(actual_data, predicted_data)
            print('MSE:', mse)
            
            report = dict(mse=mse)
        
        return report

    def serve(self, data=None):
        """
        Serve the model.
        
        Parameters
        ----------
        data : None, ``data_wrangler.DataWrangler``, numpy.array
            Data to be served on. If None, the serving data specified at the 
            wrangling stage is used.
        
        Returns
        -------
        prediction : numpy.array
            Prediction output by the model
        """

        from numpy import ravel

        # If the data is not specified, then use the serving data already 
        # specified in the wrangling stage.
        if data is None:
            data = self.data.serve        

        prediction = ravel(self.pipeline.predict(data.input))

        # TODO: Take into account the fact that ``data`` may not necessarily be
        # a ``data_wrangler.DataWrangler`` object, in which case the block 
        # below will fail.
        # 
        # If the output data has been normalized, undo the normalization so as
        # to have a more "natural" representation of the output.
        #
        # TODO: Undo any other processing of the output data
        if data.pipeline.output is not None:
            if 'normalize' in data.pipeline.output.named_steps.keys():
                if data.pipeline.output.named_steps['normalize'] is not None:
                    prediction = data.pipeline.output.named_steps['normalize'].inverse_transform(prediction)
        
        return prediction
    
    def serve_report(self):
        """
        Report on the serving. 
        """
        
        pass
    
    def run(self,
            wrangle=True,
            examine=False,
            select=False,
            train=False,
            test=False,
            serve=False):
        """
        Parameters
        ----------
        wrangle : bool
            Wrangle the data?
        examine : bool
            Examine the data?
        select : bool
            Select the model?
        train : bool
            Train the model?
        test : bool
            Test the model?
        serve : bool
            Serve the model?
            
        TODO: Take into account the possibility of passing 
        ``data_wranglers.DataWrangler`` objects instead of just boolean flags.
        """

        print('********', '*'*len(self.name), '**********', sep='*')
        print('********', self.name, '**********')
        print('********', '*'*len(self.name), '**********', sep='*')

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

class Factorizable(Problem):
    
    def __init__(
            self,
            default_args=dict(
                name='factorizable',
                report_dir='/Factorizable/reports/',
                
                # Data
                nex={'train': 200, 'test': 100},
                factor=7,
                min_int=-1000,
                max_int=1000,
                ncols=2,
                
                # Pipeline
                algo='SVC',
                n_components=None,      # PCA components
                kBest=None,             # k best natural features
                scaler={'input': True,
                        'output': True},
                params={
                    'SVC': dict(gamma=1/64),                        
                    'RFC': dict(RFC__n_estimators=35)},
                params_grid={
                    'SVC': dict(SVC__gamma=[0.0001, .001, .01, .1]),                        
                    'RFC': dict(RFC__n_estimators=[5, 10, 15, 20, 25, 30, 35])}, 
                ),
            **kwargs):
                        
        import os
        from utilities import parse_args, dict_to_dot

        kwargs = parse_args(default_args, kwargs)

        kwargs['params_grid'] = kwargs['params_grid'][kwargs['algo']]
        kwargs['params'] = dict_to_dot(kwargs['params'][kwargs['algo']])
        kwargs['nex'] = dict_to_dot(kwargs['nex'])
        
        super().__init__(**kwargs)

    def wrangle(self):
        
        from data_wranglers import Factorizable
        from utilities import dict_to_dot        
        
        self.data = dict_to_dot({
            'train': Factorizable(
                factor=self.factor,
                max_int=self.max_int,
                min_int=self.min_int,
                nex=self.nex.train,
                ncols=self.ncols),
            'test': Factorizable(
                factor=self.factor,
                max_int=self.max_int,
                min_int=self.min_int,
                nex=self.nex.test,
                ncols=self.ncols                    
                )})
            
    def pipe(self):
        
        from sklearn.pipeline import Pipeline

        if self.algo == 'SVC':

            from sklearn.svm import SVC
    
            self.pipeline = Pipeline(
                [('SVC', SVC(gamma=self.params.gamma))])
    
        elif self.algo == 'RFC':
    
            from sklearn.ensemble import RandomForestClassifier as RFC
    
            self.pipeline = Pipeline(
                [('RFC', RFC(
                    n_estimators=self.params.RFC__n_estimators))])        

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

        from utilities import dict_to_dot, parse_args

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

