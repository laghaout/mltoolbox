#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 08:34:14 2017

@author: Amine Laghaout
"""


from abc import abstractmethod
import sklearn as skl
from sklearn import metrics
import sklearn.pipeline as sklearn_pipeline
from sklearn.model_selection import GridSearchCV

from . import data_wranglers as dat
from . import utilities as util
from . import visualizers as vis


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

    into a single pipeline which can be invoked by ``run()``.

    The arguments to the ``Problem`` object as well as to all its constiuent
    functions above are passed either as a dictionary or as a string specifying
    the path to the dictionary. These arguments are processed in
    ``__init__()`` and consistency-checked by ``verify()``.

    The machine learning model itself is assembled in ``pipe()``.
    """

    def __init__(self, default_args=None, **kwargs):
        # Supersede the default arguments ``default_args`` with the arguments
        # passed as ``**kwargs``. All these  arguments will then be attributed
        # to the object.
        util.args_to_attributes(self, default_args, **kwargs)

        # Launch the chronometer for the problem.
        self.chrono = util.Chronometer()

        # Verify the consistency and integrity of the arguments.
        self.verify()

        # Assemble the model pipeline (typically with a scikit-learn pipeline).
        self.pipe()

    def verify(self):
        """
        Verify the consistency and integrity of the arguments attributed to
        the object.
        """
        pass

    def pipe(self):
        """
        Define the pipeline of estimators. This is typically where a scikit-
        learn pipeline is defined and stored as ``self.pipeline``.
        """
        pass

    @abstractmethod
    def wrangle(self):
        """
        Wrangle the data. This is where the data objects for training, testing,
        and serving could be defined and stored as, say,
        ``self.data.{train, test, serve}``.
        """

        # TODO: Create a generic wrangler and then remove the @abstractmethod.
        raise NotImplementedError

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

        # If the algorithm is a neural network in Keras, retreive the training
        # history.
        if self.algo in ['MLP', 'RNN']:
            history = self.pipeline.named_steps[self.algo].model.history

        else:
            history = None

        # Test the model on the training set.
        (data, prediction) = self.test(data)

        return (data, prediction, history)

    @staticmethod
    def train_report(data, prediction, history=None):
        """ Report on the training. """

        if history is not None:
            vis.plot_time_series(
                x=history.epoch,
                y_dict={x: history.history[x] for x in history.history.keys()},
                xlabel='epoch')

        return dict(
            history=history,
            test=Problem.test_report(data, prediction))

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

        return (data, prediction)

    @staticmethod
    def test_report(actual_data, predicted_data):
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
        except Exception:
            pass

        # Check whether the problem is a classfication, in which case the test
        # data is assessed by the confusion matrix.
        try:
            # TODO: Replace the accuracy with the confusion matrix.
            accuracy = metrics.accuracy_score(actual_data, predicted_data)
            print('Accuracy:', accuracy)

            report = dict(accuracy=accuracy)

        # If the problem is not a classification, then assume it is a
        # regression and evaluate it with the mean squared error.
        except Exception:
            mse = metrics.mean_squared_error(actual_data, predicted_data)
            print('MSE:', mse)
            report = dict(mse=mse)

        report['data'] = {'actual': actual_data, 'predicted': predicted_data}

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

        # If the data is not specified, then use the serving data already
        # specified in the wrangling stage.
        if data is None:
            data = self.data.serve

        prediction = self.pipeline.predict(data.input)

        # TODO: Take into account the fact that ``data`` may not necessarily be
        # a ``data_wrangler.DataWrangler`` object, in which case the block
        # below will fail.
        #
        # If the output data has been normalized, undo the normalization so as
        # to have a more "natural" representation of the output.
        #
        # TODO: Undo any other processing of the output data
        if data.pipeline.output is None:
            return prediction

        normed_data = data.pipeline.output.named_steps.get('normalize')
        if normed_data is not None:
            prediction = normed_data.inverse_transform(prediction)

        return prediction

    @staticmethod
    def serve_report(prediction):
        """ Report on the serving. """

        return prediction

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

        print(f'***********{"*" * len(self.name)}***********')
        print(f'********** {self.name} **********')
        print(f'***********{"*" * len(self.name)}***********')

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
            (data, prediction, history) = self.train()
            self.report['train'] = self.train_report(data, prediction, history)
            self.chrono.add_event('end train')

        if test:
            print('\n**** TEST ****\n')
            self.chrono.add_event('start test')
            (data, prediciton) = self.test()
            self.report['test'] = self.test_report(data, prediciton)
            self.chrono.add_event('end test')

        if serve:
            print('\n**** SERVE ****\n')
            self.chrono.add_event('start serve')
            prediction = self.serve()
            self.report['serve'] = self.serve_report(prediction)
            self.chrono.add_event('end serve')

#        self.chrono.view()


class BostonHousing(Problem):

    def __init__(
            self,
            default_args=dict(
                name='Boston housing prices',
                test_split=0.2,
                algo='SVR',
                params=dict(
                    SVR=dict(degree=3))),
            **kwargs):

        kwargs = util.parse_args(default_args, kwargs)

        kwargs['params'] = kwargs['params'][kwargs['algo']]

        super().__init__(**kwargs)

        print('I am in BostonHousing().')

    def wrangle(self):

        self.data = util.dict_to_dot({
            data_set: dat.BostonHousing(data_set=data_set) for
            data_set in ('train', 'test')})

    def pipe(self):

        if self.algo == 'SVR':

            self.pipeline = sklearn_pipeline.Pipeline(
                [('SVR', skl.svm.SVR(degree=self.params['degree']))])
