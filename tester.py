#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:47:51 2019

@author: ala
"""

# %% NaNs
if False:

    from utilities import rw_data
    
    A = rw_data('./AFT/data/spx_1m_stacked_2015_2016.csv', 
                parameters=dict(nrows=2000)).set_index('Dt')
    
    B = A.apply(lambda x: x.isnull().values.any(), axis=1)

# %% Alternative pipeline
if False:

    from data_wranglers import Digits
    
    print('---- Default')
    digits = Digits(encoder=False)
    digits.view()
    
    def alternative_pipe(self, n_components=37, encoder=True):

        from sklearn.pipeline import Pipeline
        from utilities import dict_to_dot

        print('Custom digits pipeline')
        
        self.input = self.raw.input.values
        self.output = self.raw.output.values

        self.n_components = n_components
        self.encoder = encoder

        pipeline = dict_to_dot({
            'input': Pipeline([
                ('reduce', self.reduce()),
                ]),
            'output': Pipeline([
                ('encode', self.encode())])})        
        
        return pipeline
    
    # New pipeline
    print('---- New pipeline')
    digits.machine_readable(alternative_pipe(digits))
    digits.view()

    # Revert back to default pipeline    
    print('---- Revert back to the default pipeline')
    digits.n_components = None
    digits.machine_readable()
    digits.view()

# %% Time series
if True:
    
    from problems import TimeSeries
    
    problem = TimeSeries()
    problem.run(train=True, test=True)
    
# %% Titanic
if False:
    
    from data_wranglers import Titanic
    
    A = Titanic()
    
    data_raw = A.raw.input
    
    print(A.raw.input.head())
    print(A.raw.output.head())

# %% Handwritten digits
if False:

    from problems import Digits

    my_problem = Digits(nex=500)

    num_rows = 500

#    my_problem.wrangle()
##    my_problem.train(my_problem.data.input[:num_rows],
##                     my_problem.data.output[:num_rows])
    my_problem.run(select=True, train=False, test=False)

# %% Synthetic classes
if False:

    from problems import SyntheticClasses

    A = SyntheticClasses(
        n_features=78,
        nex=3000,
        kBest=70,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1
    )

    A.run()

# %% Template classifier
if False:

    from data_wranglers import SyntheticClasses
    from estimators import TemplateClassifier
    from numpy import ravel
    from sklearn.metrics import accuracy_score
    from sklearn.utils.estimator_checks import check_estimator

    my_classifier = TemplateClassifier(arg_1=3)
    print(my_classifier.get_params())
    check_estimator(my_classifier)

    data = SyntheticClasses(n_features=5, n_classes=2, nex=10000)
    my_classifier.fit(data.input[:9500], ravel(data.output[:9500]))
    print('Accuracy:',
          accuracy_score(my_classifier.predict(data.input[9500:]),
                         ravel(data.output[9500:])))
