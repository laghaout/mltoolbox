#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:34:34 2017

@author: Amine Laghaout
"""

class classifier:
    
    def __init__(self, **kwargs):
        
        """
        Parameters
        ----------

        - ``self.params``: a namedtuple of model parameters

        Returns
        -------
        
        The following are naming conventions of the main attributes
        
        - ``self.train_curve``: For neural networks, this records the evolution
          of the driving metric over the epochs.
        - ``self.model``: The model (e.g., in scikit-learn, Keras, TensorFlow, 
          etc.)
        - ``self.scores``: The evaluation of the metrics
        """

        # Set the ``kwargs`` as attributes.
        [self.__setattr__(k, kwargs[k]) for k in kwargs]
        
        # Set the parameters as namedtuples.
        from collections import namedtuple
        self.params = namedtuple('params', self.params.keys())(**self.params)
        
        self.build()

    def build(self): 
        
        """
        This function builds the model. E.g., this is a neural network's 
        architecture would get implemented.
        """
        
        pass
    
    def train(self, data_object): 
        
        """
        This function trains the model.
        
        TODO:
        - Add options such as ``shuffle``, ``class_weight``, etc.
        """
        
        # Keras
        try:
        
            train_curve = self.model.fit(
                    data_object.data.input, 
                    data_object.data.output, 
                    epochs = self.params.epochs, 
                    batch_size = self.params.batch_size,
                    validation_split = self.validation_split,
                    verbose = self.verbose)

        # Scikit-learn        
        except:
            
            self.model.fit(data_object.data.input, data_object.data.output)
            
            from collections import namedtuple
            train_curve = {'history': None}
            train_curve = namedtuple('train_curve', train_curve.keys())(**train_curve)            
            
        return {'train_curve': train_curve.history}
        
    def train_report(self, report, marker = None):
        
        """
        This function provides a report on the trainig process.
        """

        from visualizers import plot2D
        
        epochs = range(1, self.params.epochs + 1)
        
        if self.params.epochs < 20 and marker is None:
            marker = 'o'

        plot2D(epochs, 
               (report['train_curve']['acc'], 
                report['train_curve']['val_acc']), 
               title = 'Model accuracy', xlabel = 'Epoch', ylabel = 'Accuracy',
               legend = ['Train', 'Test'], marker = marker)

        plot2D(epochs, 
               (report['train_curve']['loss'], 
                report['train_curve']['val_loss']),
               title = 'Model loss', xlabel = 'Epoch', ylabel = 'Loss',
               legend = ['Train', 'Test'], marker = marker)
                   
    def test(self, data_object): 
        
        """
        This function tests the model.
        """
        
        from numpy import zeros
        from pandas import DataFrame, MultiIndex
        from sklearn.metrics import confusion_matrix, f1_score
        
        # Keras
        try:
            scores = self.model.evaluate(data_object.data.input, 
                                         data_object.data.output,
                                         verbose = self.verbose)
            
            scores = {metric: scores[i] for i, metric in enumerate(self.model.metrics_names)}
            
        # Scikit-learn
        except:
            scores = {
                    'sklearn-score': self.model.score(
                            data_object.data.input, data_object.data.output)}
                    
        prediction = self.serve(data_object)

        ###

        df = DataFrame(zeros((data_object.nrows, 2*(self.output_dim + 1))))
        
        df.columns = MultiIndex.from_product(
                [['label'] + ['distribution_'+str(i) for i in range(self.output_dim)], 
                 ['actual', 'predicted']])
    
        df.loc[:, ('label', 'actual')] = data_object.data_raw[data_object.output_names].tolist()
        df.loc[:, [('distribution_'+str(x), 'predicted') for x 
                   in range(self.output_dim)]] = prediction
        df.loc[:, [('distribution_'+str(x), 'actual') for x 
                   in range(self.output_dim)]] = data_object.data.output
        df.loc[:, ('label', 'predicted')] = data_object.label_binarize.inverse_transform(prediction)

        from sklearn.metrics import accuracy_score
        scores.update({'accuracy': accuracy_score(df.label.actual, df.label.predicted)})
                
        return {'prediction': df,
                'metrics': {'scores': scores,
                            'class_names': data_object.class_names, 
                            'F1_score': f1_score(
                                    df.label.actual, 
                                    df.label.predicted, 
                                    data_object.class_names,
                                    average = 'micro'), 
                            'confusion_matrix': confusion_matrix(
                                    df.label.actual, 
                                    df.label.predicted,
                                    data_object.class_names)}}

    def test_report(self, report):    

        from visualizers import plot_confusion_matrix
                
        print('Scores:', report['metrics']['scores'])
        print('F1-score:', report['metrics']['F1_score'])
        
        plot_confusion_matrix(
                report['metrics']['confusion_matrix'], 
                classes = report['metrics']['class_names'], 
                title = 'Confusion matrix, without normalization')        
        
    def serve(self, data_object): 
        
        """
        This function serves the model.
        """
        
        return self.model.predict(data_object.data.input)
    
    def select(self, data_object):

        # Use scikit-learn to grid search the batch size and epochs
        from sklearn.model_selection import GridSearchCV
        from keras.wrappers.scikit_learn import KerasClassifier
        
        # Function to create model, required for KerasClassifier
    
        params_default = self.params
    
        # TODO: Devise a smarter way to do this.
    
        def rebuild_model(
                epochs = None, batch_size = None, loss_function = None, 
                optimizer = None, architecture = None, num_nodes = None, 
                activation = None, dropout = None):
            
            print('+++ START')
            params = dict(params_default._asdict())

            from pandas import DataFrame

            for k in self.params_space.keys():
                
                if type(self.params_space[k]) is dict:

                  for j in self.params_space[k].keys():
                      print(k, ':', eval(k), '> dict >', params[k][j])
                      params[k][j] = eval(j)
                
                
                elif type(self.params_space[k]) is type(DataFrame()):
                    
                    for j in self.params_space[k].columns:
                      print(k, ':', eval(k), '> DF >', params[k][j])
                      params[k][j] = eval(j)
                      
                else:    
                    print(k, ':', eval(k), '> direct >', params[k])
                    params[k] = eval(k)

            from collections import namedtuple
            self.params = namedtuple('params', params.keys())(**params)
            
            print(self.params)
            print('+++ END')
            
            self.build()
            
            return self.model
        
        params = dict(verbose = 0)
    
        # Grid search
        
        grid = GridSearchCV(
                estimator = KerasClassifier(build_fn = rebuild_model, **params), 
                param_grid = self.params_space, 
                n_jobs = 1, 
                verbose = 100)
        
        grid_result = grid.fit(data_object.data.input, 
                               data_object.data.output)
        
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            
        self.params = params_default

class dummy(classifier):

    def build(self):
        
        self.model = None
        
        print('Building the dummy classifier...')
        
    def train(self, data_object):
        
        self.train_curve = None
        self.input_relevance = None
        
        print('Training the dummy classifier...')
        
        return dict()
    
    def train_report(self, report, marker = None):
        
        print('Reporting on the training of the dummy classifier...')
        
        pass

    def test(self, data_object):
        
        self.scores = None
        
        print('Testing the dummy classifier...')
        
        return {'metrics': None, 
                'prediction': None}

    def test_report(self, report):
        
        print('Reporting on the testing of the dummy classifier...')
        
        pass

    def serve(self, data_object): 
        
        print('Serving the dummy classifier...')
        
        return None

class MLP(classifier):
    
    """
    Multi-layer perceptron (aka feedforward neural network)
    """
    
    def build(self):
    
        from keras.optimizers import SGD
        from keras.models import Sequential
        from keras.layers import Dense, Dropout

        self.model = Sequential()

        # Build the neural network layer by layer
        for index, layer in self.params.architecture.iterrows():
            
            # Input layer
            if index == 0:
                
                self.model.add(Dense(int(layer.num_nodes), 
                                     activation = layer.activation, 
                                     input_dim = self.input_dim))
                
                self.model.add(Dropout(layer.dropout))
            
            # Output layer
            elif index == len(self.params.architecture) - 1:

                self.model.add(Dense(self.output_dim, 
                                     activation = layer.activation))
                                
            # Hidden layers
            else:

                self.model.add(Dense(int(layer.num_nodes), 
                                     activation = layer.activation))
                
                self.model.add(Dropout(layer.dropout))
        
        # TODO: Generalize this to any kind of optimizer
        try:
            optimizer = SGD(**self.params.optimizer)
        except:
            from keras.optimizers import Adam
            try:
                optimizer = eval(self.params.optimizer)
            except: 
                optimizer = self.params.optimizer
        
        self.model.compile(loss = self.params.loss_function, 
                           optimizer = optimizer, 
                           metrics = self.metrics)
        
class RNN(classifier):

    """
    Recurrent neural network
    """
        
    def build(self):

        # LSTM with dropout for sequence classification in the IMDB dataset
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        
        
        self.model = Sequential()
        
        # With embedding
        try:
        
            from keras.layers.embeddings import Embedding
            
            # Input (embedding) layer
            self.model.add(Embedding(self.input_dim, 
                                     self.params.embed_dim, 
                                     input_length = self.max_seq_len))

            # Recurrent layer
            self.model.add(LSTM(self.params.num_nodes, 
                                dropout = self.params.dropout, 
                                recurrent_dropout = self.params.recurrent_dropout))
            
        # Without embedding
        # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        except:
                        
            # Recurrent layer
            self.model.add(LSTM(self.params.num_nodes, 
                                input_shape = (self.max_seq_len, self.input_dim),
                                dropout = self.params.dropout, 
                                recurrent_dropout = self.params.recurrent_dropout))
        
        # Output layer
        self.model.add(Dense(self.output_dim, 
                             activation = self.params.activation))
        
        self.model.compile(loss = self.params.loss_function, 
                           optimizer = self.params.optimizer, 
                           metrics = self.metrics)

class sklearn(classifier):
    
    """
    Support vector classifier
    """

    def build(self):    
    
        if self.algo == 'SVC':
            from sklearn.svm import SVC as model
        elif self.algo == 'RandomForestClassifier':            
            from sklearn.ensemble import RandomForestClassifier as model
        elif self.algo == 'ExtraTreesClassifier':
            from sklearn.ensemble import ExtraTreesClassifier as model
        
        self.model = model(**dict(self.params._asdict()))
        
    def train_report(self, report, marker = None):

        pass
    
    def select(self, data_object):
        
        from sklearn.model_selection import GridSearchCV        
        
        grid_result = GridSearchCV(self.model, self.params_space, verbose = 100)
        grid_result.fit(data_object.data.input, data_object.data.output)        

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    
    def serve(self, data_object):
        
        try:
            return self.model.predict(data_object.data.input)
        except:
            return self.model.predict(data_object)