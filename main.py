# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:52:32 2017

@author: Amine Laghaout
"""

import toolbox as tb
import classifiers as cla 
import data_wranglers as dw

(examine, select, train, test, serve) = (True, False, False, False, False)

log2file = False                # Log the errors and outputs to file?
hyperopt_algo = None            # Use the Hyperopt optimization package?
clean_script = './clean.bat'    # Delete model, report, and data
report_dir = './report/'        # Directory for the report files

# Data file for training and testing
data_files = {'dir': './data/', 'files': 'data_train.csv'}  

# Data file for serving
data_files_serve = {'dir': data_files['dir'], 'files': 'data_serve.csv'} 

features = ['f1', 'f2', 'f3']   # Feature names
batch_size = 1000               # Batch size
bunch_start = 1                 # First row of raw data to load to memory
bunch_size = 10*batch_size      # Number of rows of raw data to load to memory
nrows_train = 5*bunch_size      # Number of rows of raw data to train on
nrows_test = 5*bunch_size       # Number of rows of raw data to test on
max_seq_len = 700               # Maximum sequence length
num_epochs = 10                 # Number of epochs
num_runs = 2                    # Number of statistical runs
verbose = 2                     # Level of verbosity
max_T = 7*24*60*60              # Max. time allowed for ``select()``
max_evals = None                # Maximum dimensionality of the parameter space
metrics = ['cross_entropy', 'error'] # Metrics
save_as = tb.timestamp('YYYY-MM-DD') # Name for the model

#%% PARAMETER SPACE

from numpy import logspace
params_mesh = {'num_hidden': list(range(1, 10, 1)), 
               'learning_rate': list(logspace(-3, 1, 10)), 
               'cell_type': ['GRUCell', 'LSTMCell', 'BasicRNNCell'] , 
               'optimizer': ['AdamOptimizer', 'AdadeltaOptimizer', 
                             'AdagradOptimizer', 'GradientDescentOptimizer']}

if select:
    (params_mesh, 
     max_evals, 
     params) = cla.param_space(params_mesh, hyperopt_algo, max_evals)
else:
    params = {'num_hidden': 8,
              'learning_rate': 0.01, 
              'cell_type': 'GRUCell',
              'optimizer': 'AdamOptimizer'}

#%% BOOKKEEPING

tb.log_to_file(log2file, report_dir)

if clean_script not in {'', None}: 
    tb.runscript(clean_script, 'Running the cleaning script...')

#%% CREATE THE CLASSIFIER OBJECT.

RNN_object = cla.RNN(metrics = metrics, 
                     params = params, 
                     features = features, 
                     num_epochs = num_epochs, 
                     batch_size = batch_size, 
                     num_runs = num_runs, 
                     verbose = verbose, 
                     params_mesh = params_mesh,
                     data_files = data_files)

#%% CREATE THE DATA OBJECT.

print('\n****************** WRANGLING ******************\n')

if examine or select or train or test: 

    data_object = dw.data_wrangler(
            features = features + ['targets'], 
            bunch_start = bunch_start, 
            bunch_size = bunch_size, 
            max_seq_len = max_seq_len, 
            name = features[0], 
            nrows = nrows_train if (train or select or examine) else nrows_test, 
            verbose = True, 
            data_files = data_files)

if serve: 
   
    data_object_serve = dw.internet_case(
            features = features + ['targets'], 
            bunch_start = 1, 
            bunch_size = bunch_size, 
            max_seq_len = max_seq_len, 
            name = features[0], 
            nrows = nrows_test,
            verbose = True, 
            data_files = data_files_serve)
    
#%% EXAMINE THE DATA.

if examine:
    
    print('\n****************** EXAMINING ******************\n')
    
    data_examined = data_object.examine()
     
#%% RESTORE THE MODEL.

if select or train or test or serve:

    try:
        
        print('\n****************** RESTORING ******************\n')
    
        # Use the select/train/test data object by default for restoring the 
        # classifier. However, if it does not exist, use the data object for
        # serving the model.
        if 'data_object' not in locals():
            data_object = data_object_serve
        
        print('Attempting to restore the model...')
    
        RNN_object.restore(data_object, saved_as = save_as)
    
        # If a new verbose is specified, use it instead of that used by the 
        # restored model.
        try: RNN_object.verbose = verbose
        except: pass
    
        print('Model restored. Restored weights:')
        print(RNN_object.evaluate(data_object, 'weights')[0][0])
    
        (reinitialize, redefine) = (False, False)
        
        if select:
            select = False
            print('WARNING: Model selection will be skipped since the model already exists.')
    
#%% SELECT THE MODEL HYPERPARAMETERS.
    
    except:
        
        print('No model to restore. Gernerating it from scratch...')
        
        # Ensure that a non-existing model will next be trained.
        
        if train is False:
            train = True
            print('WARNING: The model should be trained since it does not already exist.')
            
        if select:
            
            print('\n****************** SELECTING ******************\n')     
            
            metrics_select = RNN_object.select(data_object, max_evals = max_evals)
            RNN_object.report_select(metrics_select, save_as = save_as)
    
#        RNN_object.define(data_object)
#        metrics_valid = RNN_object.validate(data_object)
#        RNN_object.report_validate(metrics_valid, save_as = save_as) 
        
        (reinitialize, redefine) = (True, True)

#%% TRAIN THE CLASSIFIER.

if train:

    print('\n****************** TRAINING ******************\n')     
    
    print('First data sample:', data_object.data_raw.iloc[0].name)
    metrics_train = RNN_object.train(data_object, 
                                     reinitialize = reinitialize, 
                                     redefine = redefine)
    RNN_object.report_train(metrics_train, save_as = save_as)
    
    RNN_object.save(save_as = save_as)
    print('Weights after saving:')
    print(RNN_object.evaluate(data_object, 'weights', num_batches = 1)[0][0]) 

#%% TEST THE CLASSIFIER.

if test:

    print('\n****************** TESTING *******************\n') 
    
    # If the data was trained, continue with the data where the training left
    # off. If not start at ``bunch_start``. 
    data_object.cue(True if train else bunch_start, nrows_test, bunch_size)
    
    print('First data sample:', data_object.data_raw.iloc[0].name)
    metrics_test = RNN_object.test(data_object)
    RNN_object.report_test(metrics_test, save_as = save_as)
    
    # Estimate how long it takes to run a hyperparameter optimization.
    RNN_object.select_runtime_estimate(
            seconds_per_trained_data_sample = metrics_train['seconds_per_data_sample'],
            seconds_per_tested_data_sample = metrics_test['seconds_per_data_sample'],
            nrows = nrows_train)
    
    # Estimate the maximum time that should be allowed to train a single data
    # sample.
    RNN_object.select_runtime_estimate(
            seconds_per_trained_data_sample = metrics_train['seconds_per_data_sample'],
            seconds_per_tested_data_sample = metrics_test['seconds_per_data_sample'],
            max_T = max_T,
            nrows = nrows_train)

#%% SERVE THE CLASSIFIER.

if serve:

    print('\n****************** SERVING *******************\n')
    
    prediction = RNN_object.predict(data_object_serve)

    print('First data sample:', data_object_serve.data_raw.iloc[0].name)
    metrics_serve = RNN_object.test(data_object_serve)
    RNN_object.report_test(metrics_serve, save_as = (save_as, 'metrics_serve'))
    
    data_raw = data_object_serve.data_raw
    data_raw['badness'] = prediction['prediction'][:,1]
    prediction = data_raw
    del data_raw
    
#%% RECORD THE OBJECT.

if select or train or test or serve:    
    RNN_object.get_settings()
