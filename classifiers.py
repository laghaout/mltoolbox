#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:39:56 2017

@author: Amine Laghaout
"""

import visualizers as vis # Graphical library

def param_space(params_mesh, hyperopt_algo = None, max_evals = None, 
                params = None):
    
    """
    This function takes the hyperparameter search space ``params_mesh`` and
    computes its dimensionality ``max_evals`` as well as processes it into a 
    format that is readily used by Hyperopt, if need be. It also extracts 
    default parameters ``params``.
    
    Parameters
    ----------
    
    params_mesh : dict of collections
        Parameter space
    hyperopt_algo : None, str
        If a string is specified with the name of a Hyperopt algorithm, then 
        generate a Hyperopt parameter space from ``params_mesh``.
    params : dict
        A single point in the hyperparameter space, typically consisting of the
        first combination from the parameter space.
    max_evals : None, str
        If ``None``, compute the dimensionality of the hyperparameter space.
    
    Returns
    -------
    
    Re-processed values of ``params_mesh``, ``max_evals``, and ``params``, if
    applicable.
    """

    # Initial parameters
    if params is None:    
        params = {param: params_mesh[param][0] for param in list(params_mesh.keys())}
    
    # Parameter space for Hyperopt
    if type(hyperopt_algo) is str:
        
        # Determine the dimensionality of the hyperparameter search space. Set 
        # That as the default number of search points when using Hyperopt.
        if max_evals is None:
            max_evals = 1                   
            for param in params_mesh.keys(): 
                max_evals *= len(params_mesh[param])
            print('The maximum number of data points in the hyperparameter search space is', max_evals)           
        
        from hyperopt import hp
        
        params_mesh = {'optimizer': hp.choice('optimizer', params_mesh['optimizer']), 
                       'num_hidden': hp.choice('num_hidden', params_mesh['num_hidden']), 
                       'learning_rate': hp.uniform('learning_rate', min(params_mesh['learning_rate']), max(params_mesh['learning_rate'])), 
                       'cell_type': hp.choice('cell_type', params_mesh['cell_type'])}

    return (params_mesh, max_evals, params)

def version_table(print2screen = True):
    
    """
    This function returns the version numbers of the various pieces of software 
    with which this module was tested.

    Notes
    -----
    In order for Hyperopt 0.1 to work, ``networkx`` had to be downgraded by
    running ``pip install networkx==1.11``. This is due to a bug that arises
    with Hyperopt when version 2.0 of ``networkx`` is installed.
    
    Parameters
    ----------
    
    print2screen : bool
        Print the version table to screen (``True``) or return it as a 
        dictionary (``False``)?
    
    Returns
    -------

    version_table : dict
        Dictionary containing the version table
    """
    
    import cpuinfo # python -m pip install -U py-cpuinfo
    import platform
    from sys import version_info
    from numpy import __version__ as np_version
    from pandas import __version__ as pd_version    
    from hyperopt import __version__ as hp_version # pip install hyperopt    
    from tensorflow import __version__ as tf_version
    from matplotlib import __version__ as plt_version
    
    version_table = {'Python': ('3.6.2', 
                                '.'.join(str(v) for v in version_info[0:3])), 
                     'TensorFlow': ('1.3.0', 
                                    tf_version),
                     'NumPy': ('1.13.3', 
                               np_version), 
                     'matplotlib': ('2.0.2', 
                                    plt_version),
                     'PyQt5': ('5.6.2', 
                               None), 
                     'pandas': ('0.21.0', 
                                pd_version), 
                     'Hyperopt': ('0.1', 
                                  hp_version), 
                     'OS': ('Linux-4.13.0-17-generic-x86_64-with-debian-stretch-sid', 
                            platform.platform()),
                     'CPU': ('Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz', 
                             cpuinfo.get_cpu_info()['brand']),
                     'GPU': ('NVIDIA GeForce GTX', 
                             None),
                     'CUDA': ('8.0.44', 
                              None)}

    if print2screen:
        
        # Maximum length of the software names
        pad = max(map(lambda x: len(x), version_table))
        
        # Print the table.
        print('software'.rjust(pad) ,': baseline', sep = '')
        print(''.rjust(pad), '  current', sep = '')
        for k in sorted(version_table.keys()):
            print(k.rjust(pad), ': ', version_table[k][0], sep = '')
            print(''.rjust(pad), '  ', version_table[k][1], sep = '')
            
    return version_table

# %% Generic classifier

class classifier:

    """
    TODO:
    
    - Save the optimal model with ``self.save()`` from within ``select()``.
    - Make sure that certain absolute metrics such as cross-entropy or loss
      are normalized by the number of examples in a batch.
    - ``eval_last_only`` should really be ``eval_best_only``. When 
      ``True``, ``record_freq`` should be set to 1.
    - Add categorical/numerical data flagging in the ``data_wrangler()``. 
      This will make it possible to subsequently use embeddings or one-hot.
    - For now, the batching only works for the processing: The entirety of
      the data is still loaded in memory. Find a way to not just process, 
      but also load the data in batches.        
    - Add dropout as a parameter for the neural networks. Should it be
      be applied via a potential ``regularize()`` method?
    - Monitor run-time compelxity with the profiler.
    - Produce UML diagrams, Sphinx-generated documentation, and a user 
      manual.
    - Add a way to systematically evaluate the baseline "naive" 
      classification so as to compare it with the learned classification.        
    - Figure out why the number of ``max_evals`` is less than the its 
      specified value: Because they are not unique.
    - ``evaluate()`` should stitch together the batches and be able to 
      distinguish the "mappings" (e.g., prediction) from the "reduces" 
      (e.g., cross-validation).
    - In ``train()``, if a model does not exist, define it first.
    - Incorporate optional statistical runs into ``train()``.
    - In ``train()``, cross-validate along with the training proper and
      record the best epoch so that we do away with the assumption that the 
      last epoch is the best.
    - Incorporate ``sklearn.feature_selection.SelectKBest()`` in 
      ``engineer_features()``.          
    - Think about how the standard deviation should be computed given the 
      nesting structure of the loops.          
    - Add a confusion matrix in the ``report_*()`` functions and break 
      down the performance by category.   
    """

    
    def __init__(self, params_mesh = None, targets = None, metrics = None, 
                 features = None, data_files = None, report_files = None, 
                 model_files = None, num_epochs = None, num_runs = None, 
                 record_freq = None, batch_size = None, verbose = None, 
                 params = None):

        """
        TODO:

        - The settings of the model should be private and only be updated via 
          functions that perform consistency checks. This will spare us the
          long block of checking and processing in the ``__init__()`` function.

        Constraints
        -----------
            
        - The metrics optimization is based on minimization. Certain metrics,
          such as accuracy should therefore be evaluated as (1 - accuracy) 
          instead. To generalize this to metrics that require maximization 
          instead, think of implementing an additional boolean variable 
          ``self.maximize``.
        - Use tuples instead of lists so as to avoid inadvertent modifications.
          E.g., ``params_mesh`` should be a tuple of dictionaries, not a list
          thereof.
        
        Parameters
        ----------
        
        params_mesh : dict of list
            Dictionary of the model parameters where each parameter is 
            associated with a list of potential values. The model will be
            evaluated with these different combinations of parameters so as to 
            find the optimal one.
        targets : str, list of str
            Names of the target features
        features : list of str
            List of the feature names
        metrics: str, list of str
            List of the metrics to be monitored. The first element of the list
            is the metric that drives the optimization. Recall that in the 
            current implementation, the driving metric is to be `minimized`.
        data_files : str, dict of str
            Dictionary specifying the directory, name, and extension of the
            data files. If a single string is provided instead, it will be 
            treated as the path name of the data file.
        report_files : dict
            Dictionary specifying the report files and directory
        model_files : dict
            Dictionary specifying the model files and directory
        num_epochs : int
            Number of epochs for training the model. (Alternatively, could we 
            use this as a ``max_iter`` limit on the number of optimization 
            iterations?)
        num_runs : int
            Number of times the model is validated. These statistical runs can 
            be used to average out fluctuations between different 
            initializations and thereby quantify them with standard deviations.
        record_freq : int
            Number of times the set of metrics is evaluated during training. 
            (Unlike the optimizer, which is based solely on the driving 
            metric, the non-driving metrics do not need to be evaluated at 
            every epoch other than for monitoring purposes.)
        batch_size : int
            Batch size of the data at each evaluation of the model.
        verbose: int, float
            Controls the frequency of prints to the stdout: 0 for complete 
            silence, 1 for warnings, 2 for reports, 3 for the statistical runs,
            4 for the K-folds, 5 for the epoch, 6 for the evaluation batches 
            Note: By using decimals, e.g., 2.5, one could also add 
            ``vis.progress_bar()``.
            
        Returns
        -------
        
        A ``classifier`` object
        """
                
        # Specify the default arguments. 
            
        if num_epochs is None: 
            num_epochs = 1
            
        if num_runs is None: 
            num_runs = 1
            
        if record_freq is None: 
            record_freq = 1
            
        if verbose is None: 
            verbose = 0 

        if params is None and type(params_mesh) is dict: 
            params = {param: None for param in params_mesh.keys()}
        elif params_mesh is None and type(params) is dict: 
            params_mesh = {param: None for param in params.keys()}
                
        if report_files is None: 
            report_files = {'dir': './report/', 'ext': '.pickle'}
            
        if model_files is None: 
            model_files = {'dir': './model/', 'ext': '.ckpt'}
            
        if data_files is None: 
            data_files = {'dir': './data/', 'files': 'data_file', 'ext': '.csv'}
        elif type(data_files) is str:
            # TODO: Generalize this to the case where there is no extension.
            directory = data_files.split('/')
            file_name = directory[-1:][0].split('.')
            directory = '/'.join(directory[:-1])+'/'
            file_extension = '.'+file_name[-1:][0]
            file_name = '.'.join(file_name[:-1])
            data_files = {'dir': directory, 
                          'files': file_name, 
                          'ext': file_extension}

        # Perform consistency checks.

        assert record_freq <= num_epochs
        
        if type(metrics) is str:
            metrics = [metrics]
            
        if type(features) is str:
            features = [features]
            
        if type(targets) is str:
            targets = [targets]
        elif targets is None:
            # TODO: Unsupervised classification
            pass    
        
        # Make sure ``params_mesh`` is a dictionary of lists
        if type(params_mesh) is dict:
            for param in params_mesh.keys():
                if type(params_mesh[param]) in [float, int, str]:
                    params_mesh[param] = [params_mesh[param]]        

        # Populate the attributes.
        
        self.params_mesh = params_mesh
        
        # Extract the sorted list of parameter names.
        if type(params_mesh) is dict:
            self.params_names = sorted(self.params_mesh.keys())
            self.opt_params = {param_name: None for param_name 
                               in self.params_names}
            
        # Determine the dimensionality of the hyperparameter search space.
        # There is a try-except block here because its body fails when 
        # using Hyperopt.
        try:
            self.params_mesh_dim = 1                   
            for param in self.params_mesh.keys(): 
                self.params_mesh_dim *= len(self.params_mesh[param])
            print('The dimensionality of the parameter search space is', 
                  self.params_mesh_dim)
        except: 
            pass            
            
        self.params = params
        self.metrics = metrics
        self.num_epochs = num_epochs        
        self.num_runs = num_runs
        self.record_freq = record_freq       
        self.batch_size = batch_size        
        self.data_files = data_files
        self.model_files = model_files
        self.report_files = report_files                
        self.features = features
        self.targets = targets
        self.verbose = verbose
        
        # Specify whether the optimization consists of a maximization or not.
        # TODO: Find a more consistent way to do this. Should everything be 
        #       passed as an argument to the constructor?
        self.maximize = False
    
    def train(self, data_object, eval_last_only = False, redefine = True, 
              reinitialize = True):
        
        """
        Parameters
        ----------
        
        data_object : Object of class ``data_wrangler()``
            Training data
        eval_last_only : bool
            Decide whether to record the training metrics for every
            ``self.record_freq`` epochs or at a single epoch only (by default,
            the last one).
        redefine : bool
            Redefine the model based on its current parameters?
        reinitialize : bool
            Reinitialize the model?            

        Returns
        -------
        
        metrics : dict
            Dictionary of the training metrics
        """

        if redefine:
            # TODO: In general, ``define()`` does not need to take in the data.
            #       This is only the case here because the inherited RNN
            #       class needs to know the dimensions of the data.
            #       Find a more generalizable call to ``define()`` without
            #       ``data_object`` as an argument.
            self.define(data_object)
            
        if reinitialize:
            self.initialize()

        import time
        from math import ceil
        from numpy import zeros, array, uint32, argmin # , inf
        
        # Distinguish between the baseline and non-baseline metrics as only the 
        # latter need to be evaluated at each ``record_freq`` epochs.
        non_BL_metrics = [m for m in self.metrics if m[:3] != 'BL_']
        BL_metrics = [m for m in self.metrics if m[:3] == 'BL_']
        
        # Determine the number of times the metrics shall be recorded.
        if eval_last_only:
            num_metrics_points = 1
        else:
            num_metrics_points = ceil(self.num_epochs/self.record_freq)
        
        assert 0 <= num_metrics_points <= self.num_epochs
        
        # Initialize the dictionary of metrics to ``None``.
        metrics = {str(m): [None]*num_metrics_points for m in self.metrics}
        
        # Make sure the number of epochs can be represented as a uint32. 
        assert 1 <= self.record_freq <= self.num_epochs < 2**32
        
        # Array of the epochs at which the metrics shall be recorded
        metrics['epoch'] = zeros(num_metrics_points, dtype = uint32)
              
        # Counter for the number of metric recordings.
        index = 0

        # Optimal value of the driving metric over all epochs
#        opt_driving_metric = (-1)**self.maximize*inf 

        # Start to record the average time per epoch.
        metrics['seconds_per_data_sample'] = time.time()

        # For every epoch...
        for epoch in range(self.num_epochs):

            if self.verbose == 4.5:
                vis.progress_bar(epoch, self.num_epochs)
            elif self.verbose >= 5:
                print('  ----------------- EPOCH', epoch)                
            
            # Optimize the driving metric.
            self.optimize(data_object)
            
            # TODO: Check for the ``best_epoch`` but evaluate it on the 
            #       `validation` set, not the training set as is done here.
            # TODO: Factor in ``self.maximize`` to account for optimizations 
            #       based on maximization.            
#            curr_driving_metric = self.evaluate(data, self.metrics[0])[0]
#            if curr_driving_metric < opt_driving_metric:
#                metrics['best_epoch'] = epoch
#                opt_driving_metric = curr_driving_metric
            
            # Should we record the metrics at every ``self.record_freq`` 
            # epochs?
            record_every = epoch % self.record_freq == 0 and eval_last_only is False
            
            # Should we record the metrics at the last epoch only?
            # TODO: Once ``eval_last_only`` is converted to ``eval_best_only``,
            #       rename ``record_last`` to ``record_best``.
            record_last = epoch == self.num_epochs - 1 and eval_last_only is True
            
            # If either one of the two mutually exclusive conditions above is
            # true, record the metrics at the current epoch. Only do so for the 
            # non-baseline metrics
            if record_every or record_last:

                metrics_evaluation = self.evaluate(data_object, non_BL_metrics)
                
                # Assign the recorded metrics in the order they appear in the
                # list ``self.metrics``.
                # TODO: We need a more robust way to do this without relying
                #       on the assumptions that ``metrics_evaluation`` contains
                #       the evaluations of the metrics as they appear in
                #       ``self.metrics``. This could be done with a dictionary
                #       for example. Cf. ``evaluate()``.
                for k, metric in enumerate(non_BL_metrics):
                    metrics[metric][index] = metrics_evaluation[k]
                
                metrics['epoch'][index] = epoch
                
                index += 1
                
#                #---------------------------
#                key_metric = 
#                metrics['opt_epoch'] = argmin(metrics[self.metrics[0]])
#                #---------------------------

        # Now that we are out of the epoch loop, record the baseline metrics, 
        # if any.
        for k, metric in enumerate(BL_metrics):
            if metric[:3] == 'BL_':
                metrics[metric] = self.evaluate(data_object, metric)
        
        # For the sake of consistency, convert all the lists to arrays.
        for metric in metrics.keys():
            if type(metrics[metric]) is list:
                metrics[metric] = array(metrics[metric])

        # Compute the mean running time per data sample per epoch.
        metrics['seconds_per_data_sample'] = time.time() - metrics['seconds_per_data_sample']
        metrics['seconds_per_data_sample'] /= (self.num_epochs*data_object.nrows)

        # TODO: Use ``self.maximize`` to determine whether one should use 
        #       ``argmin`` or ``argmax``.
        metrics['opt_epoch'] = argmin(metrics[self.metrics[0]])

        return metrics

    def report_train(self, metrics, dec_digits = 10, save_as = None, 
                     data = None):
        
        """
        This function summarizes the evolution of the metrics per training 
        epoch.
        
        TODO:

        - Make it possible to plot other types of metrics which are of higher
          dimensions, e.g., a cascaded sequence of 2D color histograms for the
          weights of a neural network.
            
        Parameters
        ----------

        metrics : dict
            Dictionary of the metrics evaluated at training
        dec_digits: int
            Number of decimal digits for displaying the metrics
        save_as : str, True, None
            Path name for the plots to be saved. Use ``None`` to not save. Use
            ``True`` to save in the default report directory with the 
            auto-generated name produced by ``metadata_tag()``.
        data : dict of pandas.DataFrame, dict of numpy.array
            Training data. When not ``None``, the argument ``metrics`` is 
            ignored.
        """

        # If no data is provided, do the training first.
        if data is not None:
            metrics = self.train(data)

        if self.verbose >= 2:

            print('\n***** TRAIN REPORT\n')
            
            for k in sorted(metrics.keys()):
                
                # If the ``value`` is a single-element array, get that element
                # only. Otherwise, get the best epoch only, which we assume to 
                # be the last epoch for now.
                # TODO: Figure out a more elegant way to avoid this and 
                #       generalize for the case when the best epoch is not 
                #       necessarily the last one.
                value = metrics[k]
                try: value = value[-1] # TODO: Instead of -1, use ``metrics['best_epoch']``
                except: pass
                print('  - ', k, ': ', value, sep = '')
        
            if self.verbose >= 3:
                print('\n***** END TRAIN REPORT\n')
                
        for metric in self.metrics:
            
            # TODO: For now, we only plot these one-dimensional metrics, but 
            # generalize  this for any kind of tensors, including those that 
            # are not 2D.
            #
            # Also, do not plot if the metric is a baseline. I.e., if it starts
            # with ``BL_``.
            if len(metrics[metric].shape) == 1 and metric[:3] != 'BL_':
                
                plot_path = self.save_labeled(
                        None, save_as, './report/', metric+'_vs_epoch', suffix = '.pdf')
                
                vis.plot2D(metrics['epoch'], 
                           metrics[metric], 
                           xlabel = 'epoch', 
                           ylabel = metric, save_as = plot_path)  
        
        # Save the metrics.
        self.save_labeled(metrics, save_as, './report/', 'metrics_train')

    def select(self, data_object, max_evals = None, algo = 'tpe'):
        
        """
        This function performs the model selection based on the different 
        combinations of model parameters ``self.params_mesh``.
        
        Pre-selection checklist
        -----------------------
        
        The hyperparameter selection process is very time consuming and it is
        therefore important to get it right before launching it. This can be
        done by going over the following checklist (if applicable):
            
            - Choose the right search space for the hyperparameters. This often
            means that the sampling has to be done over a logarithmic scale.
            - Determine how much can be loaded into memory. Set 
            ``data_object.bunch_size`` accordingly. Make the data about 75% of 
            the maximum capicity so as to allow for extra RAM to be used for 
            other processes.
            - Select the most time consuming hyperparameter (e.g., the highest
            number of hidden units in a neural network).
            - Adjust the hyperparameters so as meet the desired runtime for
            ``select()``. Use ``select_runtime_estimate()``.          
        
        Parameters
        ----------
        
        data_object : Object of class ``data_wrangler()``
            Training data
        max_evals : int, None
            Maximum number of evaluations using Hyperopt. If ``None``, use 
            a grid search instead of Hyperopt.
        algo : str
            Algorithm to use when using Hyperopt.
            
        Returns
        -------
        
        metrics : dict
            Dictionary summarizing the metrics at all the parameter runs.
        """

        import time  
        
        delta_t = time.time()

        if self.verbose >= 2: 
            print('\n***** MODEL SELECTION\n')
        
        # Initialize the optimum value of the driving metric.
        from numpy import inf
        global metric_optimum 
        metric_optimum = (-1)**self.maximize*inf
        
        # Dictionary storing the training and validation metrics for each of 
        # the parameter combinations.
        metrics = {}
        
        # If ``max_evals`` is ``None``, detect whether the parameters mesh 
        # contains Hyperopt search spaces. If so, set ``max_evals`` to a 
        # default number, namely twice the number of parameters.
        # TODO: Review the validity of this.
        if 'Apply' in [type(param).__name__ for param in self.params_mesh.values()] and max_evals is None:
            max_evals = 2*len(self.params_names)
            
        # If ``max_evals`` is inadvertently set to a specific value while the 
        # parameters mesh does not contains Hyperopt search spaces, reset 
        # ``max_evals`` to ``None``.
        elif 'Apply' not in [type(param).__name__ for param in self.params_mesh.values()] and max_evals is not None:
            max_evals = None
        
        # NOTE: ``metric_optimum`` is declared as global so that it can be 
        #       accessed within ``validate_within_hyperopt()`` below. 
        
        # If ``max_evals`` is ``None``, use the complete grid search.
        if max_evals is None: 
            
            print('Launching grid search...')
            
            # Generate all the possible tuples of model parameters.
            import itertools
            params_values = [self.params_mesh[param] for param 
                             in self.params_names]
            params_combinations = list(itertools.product(*params_values))

            # For each combination of the model parameters...
            for params_combination in params_combinations:
                
                # Set the model parameters to be the current combination.
                for i, param in enumerate(self.params_names):
                    self.params[param] = params_combination[i]
    
                # TODO: Once again, in the general case, ``define()`` is not 
                #       supposed to know the data. We only do this here because 
                #       we know that we inherit the class for the RNN and that 
                #       the RNN in TensorFlow needs to know the data 
                #       dimensions. 
                #       Generalize this so that ``define()`` does not need to 
                #       know the data. This could be done by adding new 
                # attributes (e.g., ``self.data_dims`` for example).
                self.define(data_object)

                # Retrieve the training and validation metrics for the current
                # combination of parameters.
                metrics[params_combination] = self.validate(data_object)
    
                # Record the optimum combination of model parameters if the 
                # driving metric reaches a new optimum on the validation set.
                # TODO: Flip the inequality depending on the potential boolean 
                #       toggle ``self.maximize``.
                if metrics[params_combination]['valid'][self.metrics[0]] < metric_optimum:                
                    self.opt_params = self.params.copy()
                    metric_optimum = metrics[params_combination]['valid'][self.metrics[0]]
        
        # If ``max_evals`` is not ``None``, use Hyperopt.
        else: 
            
            print('Launching hyperopt search...')
            
            from hyperopt import fmin, tpe, rand, STATUS_OK, Trials
                        
            def validate_within_hyperopt(args):
                
                global metric_optimum
                
                # Set the current parameters of the model.
                for param in self.params.keys():
                    self.params[param] = args[param]
                               
                self.define(data_object)
                                
                # Tuple of the current combination of parameters, ordered as in
                # ``self.params_names``.
                params_combination = tuple([self.params[param] for param 
                                            in self.params_names])
                
                # Perform the validation with the current combination of 
                # parameters...
                metrics[params_combination] = self.validate(data_object)
                
                # ... and retrieve its performance based on the driving metric.
                loss = metrics[params_combination]['valid'][self.metrics[0]]
                
                # Record the optimum combination of model parameters if the 
                # driving metric reaches a new optimum on the validation set.
                # TODO: Flip the inequality depending on the potential boolean 
                #       toggle ``self.maximize``.
                if metrics[params_combination]['valid'][self.metrics[0]] < metric_optimum:                
                    self.opt_params = self.params.copy()
                    metric_optimum = metrics[params_combination]['valid'][self.metrics[0]]                
                                
                return {'loss': loss, 'status': STATUS_OK}

            # TODO: Ensure we have distinct trials.
            trials = Trials()
                        
            best = fmin(
                    fn = validate_within_hyperopt, 
                    space = self.params_mesh, 
                    algo = eval(algo+'.suggest'), 
                    max_evals = max_evals, 
                    trials = trials)
    
            metrics['trials'] = trials
            metrics['best'] = best
                        
        # Update the parameters with the optimal parameters. 
        self.params = self.opt_params.copy()

        if self.verbose >= 3:
            print('\n***** END MODEL SELECTION\n')

        delta_t = time.time() - delta_t
        
        metrics['delta_t'] = delta_t

        return metrics

    def select_runtime_estimate(self, 
            seconds_per_trained_data_sample = None,
            seconds_per_tested_data_sample = None,
            nrows = 1, 
            num_K_folds = 3,
            max_T = None):
        
        """
        This function estimate the time it takes to perform the hyperparameter
        optimization on neural networks.
        
        Note that this is only a rough approximation since the times in 
        training a data sample depend on the how many metrics are computer per 
        epoch (see ``train()``).
        
        Parameters
        ----------
        
        seconds_per_trained_data_sample : float
            Average number of seconds require to train on a single data sample 
            per epoch
        seconds_per_tested_data_sample : float
            Average number of seconds require to test on a single data sample 
        nrows : int
            Number of data samples for the training set, i.e., training proper
            and cross-validation sets
        num_K_folds : int
            Number of K-folds for cross-validation
        max_T : bool, float
            If set to a float, the function returns the maximum time that 
            should be allowed for training a single data sample if ``max_T`` is 
            the maximum time we allow for hyperparameter optimization.
        
        Returns
        -------
        
        T : float
            Number of seconds expected to perform hyperparameter optimization,
            or, if ``max_T`` is specified, the maximum number of seconds 
            allowed per training of a data sample so as to have the hyper-
            parameter optimization taken time ``max_T``.
        """

        num_epochs = self.num_epochs
        num_runs = self.num_runs
        dim_search_space = self.params_mesh_dim
        
        # If the runtime to test a data sample is not specified, just assume it is
        # the same as the time to train a data sample. (This will of course give
        # an upper bound.)
        if seconds_per_tested_data_sample is None:
            seconds_per_tested_data_sample = seconds_per_trained_data_sample

        if type(max_T) in {float, int}:
            
            T = max_T/(((num_K_folds - 1)*num_epochs + 1)*nrows*num_runs*dim_search_space)
            print('The maximum allowed runtime per data sample is [assuming the training ~ testing]')
            
        else:

            T = (num_K_folds - 1)*seconds_per_trained_data_sample*num_epochs + seconds_per_tested_data_sample
            T *= nrows*num_runs*dim_search_space            
            print('The hyperparameter selection process will take')
        
        print(T, 'seconds, i.e.,', T/3600, 'hours, or', T/(3600*24), 'days.')
            
        return T

    def validate(self, data_object):
        
        """
        This function performs the statistically averaged, K-folded, cross-
        validation of the model given the parameters ``self.params``.
        
        Parameters
        ----------
        
        data_object : Object of class ``data_wrangler()``
            Training data
        
        Returns
        -------
        
        metrics : dict
            Dictionary of the training and validation metrics averaged over the 
            K-folds.
            
            - ``metrics['train']`` are the metrics from the best epoch of the 
              training set.
            - ``metrics['valid']`` are the metrics from the  validation set.        
        """
        
        from numpy import mean, std

        metrics = {'train': {}, 'valid': {}}

        # Lists of metrics over the different statistical runs
        train_metrics_run = []
        valid_metrics_run = []

        # Save the current settings of the data object so as to restore them
        # once we are done with the batch loops.
        bunch_start = data_object.bunch_start
        nrows = data_object.nrows
        bunch_size = data_object.bunch_size
        
        # For each statistical run...
        for r in range(self.num_runs):
            
            if self.verbose >= 3:
                print('================ RUN ', r, '/', self.num_runs - 1, 
                      sep = '')
            
            # Lists of metrics over the different K-folds
            train_metrics_fold = []
            valid_metrics_fold = []
            
            # Index of the K-folds
            fold_index = 0

            for train_index, test_index in data_object.split(stratify = False): # Should eventually allow for stratification

                if self.verbose >= 4:
                    print(' ________________ FOLD ', fold_index, '/',                           
                          data_object.KFolds.n_splits  - 1, sep = '')
                
                data_object.cue(bunch_start = test_index, 
                                nrows = nrows - len(test_index))
                
                if self.verbose >= 4:
                    print('  ++++++++++++++++ TRAIN')

                # Each K-fold needs to be initialized separately.
                train_metrics_fold.append(
                        self.train(data_object, 
                                   eval_last_only = True, # TODO: Replace with ``eval_best_only = True``.
                                   redefine = False, 
                                   reinitialize = True))
                
                # Restore
                data_object.cue(bunch_start, nrows, bunch_size)
                
                if self.verbose >= 4:
                    print('  ++++++++++++++++ CROSS-VALIDATE')
                
                data_object.cue(bunch_start = train_index, 
                                nrows = nrows - len(train_index),
                                bunch_size = min(nrows - len(train_index), data_object.bunch_size))
                
                valid_metrics_fold.append(self.test(data_object))
                
                # Restore
                data_object.cue(bunch_start, nrows, bunch_size)
                
                fold_index += 1

            # Average the results from the K-folding.
            # NOTE: We only keep track of the standard deviation between
            #       statistical runs, not between K-folds.
            # TODO: Does it make more sense to compute the means and 
            #       standard deviations over both statistical runs and 
            #       K-folds??
            
            for k in train_metrics_fold[0].keys():
                metrics['train'][k] = mean(
                        [fold[k] for fold in train_metrics_fold], axis = 0)

            train_metrics_run.append(metrics['train'].copy())

            for k in valid_metrics_fold[0].keys():
                metrics['valid'][k] = mean(
                        [fold[k] for fold in valid_metrics_fold], axis = 0)
            
            valid_metrics_run.append(metrics['valid'].copy())

        # Average the results from the different statistical runs.

        for k in train_metrics_run[0].keys():
            metrics['train'][k] = mean(
                    [run[k] for run in train_metrics_run], axis = 0)
            metrics['train'][k+'_std'] = std(
                    [run[k] for run in train_metrics_run], axis = 0)
        
        for k in valid_metrics_run[0].keys():                 
            metrics['valid'][k] = mean(
                    [run[k] for run in valid_metrics_run], axis = 0)
            metrics['valid'][k+'_std'] = std(
                    [run[k] for run in valid_metrics_run], axis = 0)                

        if self.verbose >= 2:
            print(self.params, ': ', self.metrics[0], ' = ', 
                  metrics['valid'][self.metrics[0]], sep = '')

        return metrics

    def report_select(self, metrics, dec_digits = 10, save_as = None):
        
        """
        This function prints a summary table or image of the driving metrics 
        for the various models evaluated on the parameter search space
        ``self.params_mesh``. It then calls ``report_validate()`` to present 
        the validation metrics from the best-performing model.
        
        Parameters
        ----------
        
        metrics : dict
            Dictionary of metrics at training and validation for each of the 
            parameter combinations in the mesh.
        dec_digits : int
            Number of decimal digits for displaying the metrics
        """

        # Save the summary image under the name generated by 
        # ``metadata_tag()``.
        
        plot_path = self.save_labeled(
                None, save_as, './report/', self.metrics[0]+'_param_space', suffix = '.pdf')

        # TODO: ``self.params_mesh`` should be replaced with 
        #       ``self.params_mesh_dim`` when using Hyperopt.
        if len(self.params_mesh) == 1:
            vis.plotParamsSpace(
                    (self.params_mesh[self.params_names[0]]),
                    self.metrics[0],
                    metrics, 
                    xlabel = self.params_names[0],
                    save_as = plot_path)
        elif len(self.params_mesh) == 2:
            vis.plotParamsSpace(
                    (self.params_mesh[self.params_names[0]], 
                     self.params_mesh[self.params_names[1]]), 
                    self.metrics[0],
                    metrics, 
                    xlabel = self.params_names[1],
                    ylabel = self.params_names[0],
                    save_as = plot_path)                        
        elif len(self.params_mesh) > 2:
            if self.verbose >= 1:
                print('WARNING: report_select() cannot plot in more than 2D for now.')
            pass
        
        if 'trials' in metrics.keys():
            for param in self.params_names:
                
                plot_path = self.save_labeled(
                        None, save_as, './report/', param+'_trials', suffix = '.pdf')
                
                vis.visualize_learning(param, metrics['trials'], 
                                       save_as = plot_path)
        
        # TODO: Print a warning if the optimal parameters are at the edges of 
        # search range (and then re-centre the mesh so as to prepare for a new
        # parameter selection?)
        
        if self.verbose >= 2:
            
            print('\n***** SELECTION REPORT\n')
            
            print('The optimal set of parameters based on the metric \'', 
                  self.metrics[0], '\' is:', sep = '')
            
            for opt_param in sorted(self.opt_params.keys()):
                
                print('  - ', opt_param, ': ', self.opt_params[opt_param], 
                      sep = '')
            
            opt_params_values = tuple([self.opt_params[param] for param 
                                       in self.params_names])
            
            print('\nThe validation metrics for this optimal set of parameters are:')
            self.report_validate(metrics[opt_params_values], save_as = save_as)
                
            if self.verbose >= 3:
                print('***** END SELECTION REPORT\n')
        
        # Save the metrics.           
        self.save_labeled(metrics, save_as, './report/', 'metrics_select')        

    def report_validate(self, metrics, dec_digits = 10, save_as = None):
        
        """
        This function summarizes the metrics from the validation.
        
        Parameters
        ----------
        
        metrics : dict
            Dictionary of the metrics evaluated at training and validation
        dec_digits : int
            Precision in decimal digits when printing out numbers
        """
        
        if self.verbose >= 2:
            
            print('\n***** VALIDATION REPORT\n')

            print('  - Training set (optimal epoch):')
            for k in sorted(metrics['train'].keys()):

                # If the ``value`` is a single-element array, get that element
                # only.
                # TODO: Find a more elegant way to do this.
                value = metrics['train'][k]                
                try: value = value[-1]
                except: pass
                print('      - ', k, ': ', value, sep = '')
                
            print('  - Validation set:')            
            for k in sorted(metrics['valid'].keys()):
                print('      - ', k, ': ', metrics['valid'][k], sep = '')
                
            if self.verbose >= 3:
                print('\n***** END VALIDATION REPORT\n')
                
        # Save the metrics.           
        self.save_labeled(metrics, save_as, './report/', 'metrics_validate') 

    def test(self, data_object):

        """
        This function evaluates the metrics on ``data``.
        
        Parameters
        ----------
        
        data_object : Object of class ``data_wrangler()``
            Training data
        
        Returns
        -------
        
        metrics : dict
            Dictionary of the metrics evaluated at testing
        """

        import time          

        metrics = {}
        metrics['seconds_per_data_sample'] = time.time()
        metrics_evaluation = self.evaluate(data_object)        

        # Retrieve the metrics in the order they were evaluated, i.e., the same
        # order in which they appear in the list ``self.metrics``. This is an 
        # important assumption which is made throughout.
        # TODO: Use a dictionary as a more robust alternative to the list.
        for k, metric in enumerate(self.metrics):            
            metrics[metric] = metrics_evaluation[k]

        metrics['seconds_per_data_sample'] = (time.time() - metrics['seconds_per_data_sample'])/data_object.nrows

        return metrics
    
    def report_test(self, metrics, save_as = None, dec_digits = 10, ):
        
        """
        This function summarizes the metrics evaluated from the testing set.
        
        Parameters
        ----------
        
        metrics : dict
            Dictionary of the metrics evaluated at testing
        dec_digits: int
            Precision in decimal digits when printing out numbers            
        """

        filename = 'metrics_test'
        if type(save_as) is tuple:
            filename = save_as[1]
            save_as = save_as[0]

        if self.verbose >= 2:
            
            print('\n***** TEST REPORT\n')
            
            for k in sorted(metrics.keys()):
                value = metrics[k]
                print('  - ', k, ': ', value, sep = '')

            if self.verbose >= 3:
                print('\n***** END TEST REPORT\n')
                            
        # Save the metrics.           
        self.save_labeled(metrics, save_as, './report/', filename)             
            
    def identifier(self, name = None, prefix = '', suffix = ''):
        
        name = self.metadata_tag() if name is None else name
        
        return prefix+name+suffix
            
    def save_labeled(self, 
                     object_to_save = None, 
                     save_under = None, 
                     base_dir = './', 
                     name = None, 
                     prefix = '', 
                     suffix = '.pickle'):
        
        """
        This function saves an object with the labeling produced specifically
        for the `self` object.
        """

        import os
        from pickle import dump
        
        sub_dir =  base_dir + self.identifier(save_under)+'/'
        save_as = sub_dir + self.identifier(name, prefix, suffix)
        
        os.makedirs(os.path.dirname(save_as), exist_ok = True)
        
        if object_to_save is not None and save_as is not False:
            dump(object_to_save, open(save_as, 'wb'))
            
        return save_as

    def save(self, temp_self, save_as = None, prefix = 'model_object'):
        
        """
        This function saves the model object.
        """
        
        from pickle import dump
        
        self.model_files['dir'] =  './model/'+self.identifier(save_as)+'/'
        
        save_as = self.model_files['dir'] + self.identifier(
                '', prefix = prefix, suffix = '.pickle')
        
        dump(temp_self, open(save_as, 'wb'))

    def restore(self, saved_as = None, prefix = 'model_object'):    
        
        from pickle import load
        
        self.model_files['dir'] =  './model/'+self.identifier(saved_as)+'/'
        
        saved_as = self.model_files['dir'] + self.identifier(
                '', prefix = prefix, suffix = '.pickle')
        
        restored_self = load(open(saved_as, 'rb'))  
        
        # TODO: Use `self.set_params()` instead. Also consider using 
        #       `self.__dict__ = restored_self.__dict__.copy()` instead.
        #       Cf. <https://stackoverflow.com/questions/47421289/assigning-an-oop-object-to-another>
        for attribute in restored_self.__dict__.keys():
            self.__setattr__(attribute, restored_self.__getattribute__(attribute))
            
        return saved_as
    
    def predict(self, data_object, metrics2predict = 'prediction'):
        
        """        
        Predict (i.e., infer), the targets given the features.

        TODO:
            
        - Account for the case when the data is shorter than 
          ``self.batch_size``. This may need the incorporation of 
          ``batch_data`` here, but may not work with TensorFlow due to its
          rigidity with batch sizes for RNNs.
        
        Parameters
        ----------

        data_object : Object of class ``data_wrangler()``
            Training data
        metrics2predict : list, str
            Name of the metrics to predict
            
        Returns
        -------
        
        evaluated_predictions : dict
            Dictionary of the metrics and their predictions concatenated by 
            batches
        """
        
#        data = data_object.data
        
        from numpy import vstack

        if type(metrics2predict) is str:
            metrics2predict = [metrics2predict]

#        num_batches = int(len(data['features'])/self.batch_size)
#        batch_start = 0
#        batch_end = self.batch_size

        # Evaluate the first batch.
        evaluated_predictions = {
                metric: self.evaluate(data_object,
                         metric)[0] # [0] because ``evaluate()`` returns a list
                for metric in metrics2predict}
        
#        # Stitch together the rest of the batches.
#        while num_batches > 1:
#            
#            num_batches -= 1
#            batch_start += self.batch_size
#            batch_end += self.batch_size
#            
#            evaluated_predictions = {
#                    metric: vstack(
#                            (evaluated_predictions[metric],
#                             self.evaluate(
#                                     {'features': data['features'][batch_start:batch_end],
#                                     'seq_len': data['seq_len'][batch_start:batch_end]},
#                                     metric)[0]))
#                    for metric in metrics2predict}
                    
        return evaluated_predictions
    
    def report_predict(self, 
                       prediction, 
                       input_data,
                       seen_data = None, 
                       output_target_columns = None):

        """
        This function helps compiles the results of a ``prediction`` from
        input data ``input_data``. It also states whether the data was already
        seen in training, i.e., if it appeared in ``seen_data``.
        
        Parameters
        ----------
        
        prediction : dict
            Dictionary whose entry ``prediction`` contains the predictions of 
            the model
        input_data : pandas.DataFrame
            Data frame that contains the raw input data from which we want to
            evaluate predictions.
        seen_data : pandas.DataFrame
            Data frame of the raw data that was used for the training
        output_target_columns : list
            Column names of the predicted targets
        
        Retrurns
        --------
        
        predict_table : pandas.DataFrame
            Data frame that concatenates ``input_data``, whether the input data
            was part of ``seen_data``, and the predicted values ``prediction``.
        """

        predict_table = input_data
        try:
            predict_table['seen'] = [True if x in seen_data.index else False 
                         for x in input_data.index]
        except:
            pass
        for x, output_target in enumerate(output_target_columns):
            predict_table[output_target] = prediction['prediction'][:, x]

        return predict_table
    
    def define(self):
        
        """
        Define the classifier.
        
        TODO:
        
        - Finish here or inherit.       
        """
        
        pass 
    
    def initialize(self):
        
        """
        Initialize the classifier. This is relevant when there is a random
        initialization to be performed on the model parameters.

        TODO:
        
        - Finish here or inherit.   
        """
        
        pass 
    
    def optimize(self, data_object, driving_metric = 'optimize'):
        
        """
        Optimize the parameters of the model. For example, this is where 
        gradient descent will typically go.
        
        Parameters
        ----------
        
        data_object : Object of class ``data_wrangler()``
            Data over which the optimization is performed    
        
        driving_metric : str 
            Name of the principal metric, i.e., the one which drives the 
            optimization.
            
        Returns
        -------
        
        optimized_driving_metric : TBD
            Whichever value the optimizer takes on once evaluated
        """
        
        if self.verbose >= 6:
            print('  - Optimizing', driving_metric)
        
        optimized_driving_metric = self.evaluate(data_object, driving_metric)
        
        return optimized_driving_metric    
    
    def evaluate(self, data_object, metrics = None):
        
        """       
        Evalute the metrics on the data set ``data``.

        TODO:
        
        - Finish here or inherit.
        - Perform the evaluation in batches.
        
        Parameters
        ----------
        
        data_object : ``data_wrangler()`` object
            Data over which the model is evaluated
        
        metrics : list of str
            List of metrics or operations to be evaluated. If set to ``None``,
            use ``self.metrics`` instead.
        """

        if metrics is None:
            metrics = self.metrics
        elif type(metrics) is str:
            metrics = [metrics]
            
        pass

    def get_settings(self, show = None, no_show = set()):
        
        """
        Print all the attributes of the object.
        
        show : list, None
            TODO: As an option, show only:
                
            - Hyperparameters
            - Book-keeping
            - Files and directories
            - Metrics and driving metric
        """
        
        from toolbox import get_object_settings
        
        get_object_settings(self, show, no_show)
    
    def metadata_tag(self, prefix = '', suffix = '', alt_name = None):
        
        """
        This function returns the name which helps identify a certain model or 
        report based on the parameters and settings that are associated with
        it. An alternative name ``alt_name`` can also be specified instead. 
        A ``prefix`` and ``suffix`` can be added to the name.
        
        Parameters
        ----------
        
        prefix : str
            String to potentially prefix to the model name
        suffix : str
            String to potentially suffix to the model name
        alt_name : str
            Alternative name specified externally
            
        Returns
        -------
        
        name : str
            Name for the report or model
        """
        
        if alt_name is None:
        
            if prefix != '':
                name = '_'
            else:
                name = ''
                
            for param_name in sorted(self.params.keys()):
                name += param_name+'_'+str(self.params[param_name])+'_'
                
            name += 'metric_'+str(self.metrics[0])
            name += '_num_epochs_'+str(self.num_epochs)
            name += '_num_runs_'+str(self.num_runs)
            name += '_batch_size_'+str(self.batch_size)

            if suffix != '':
                name += '_'
        
        else:
            
            name = str(alt_name)
                
        name = str(prefix)+name+str(suffix)
        
        return name

    def custom(self, external_function = None):
        
        """        
        This function allows an external function, i.e., one defined outside of
        this class, to access all the attributes of the class and thereby 
        behave as if it were one of its methods. This may come in handy when,
        for example, a custom data wrangling function may need to be defined.
        
        Parameters
        ----------
        
        external_function : tuple
            Tuple where the first element is a function defined externally and
            the remaining elements are its parameters.

        TODO:
            
        - Consider using a dictionary instead of a tuple. That will make it 
          easier to label the arguments.
            
        Returns
        -------
            
        Output of the external function, if any.
        
        Example
        -------
        
        >>> def foo(self, name, age, country):
        >>>     print('Hello there,', name)
        >>>     print('You are', age, 'years old')
        >>>     print('You are from', country)
        >>>     
        >>> def foo2(self):
        >>>     print('This function takes no argument, but can change ``self`` values.')
        >>>     self.some_attribute = 'New value for the attribute'        
        >>> 
        >>> import classifiers as cla 
        >>> myClassifierObject = cla.classifier()
        >>> 
        >>> myClassifierObject.wrangle_data((foo, 'Olof', 36, 'Sweden'))
        >>> myClassifierObject.wrangle_data((foo2, ))
        >>> print(myClassifierObject.some_attribute)     
        """
        
        assert type(external_function) is tuple
        
        return external_function[0](self, *external_function[1:])

    def score(self, data):

        """
        For the sake of scikit-learn
        TODO: Delete if useless.
        
        Score function needed by scikit-learn for parameter search
        
        Parameters
        ----------
        
        data : dict of pandas.DataFrame, dict of numpy.array
            Training data including both training proper and validation
        
        
        Returns
        -------
        
        The evaluation of the driving metric
        """

        return self.evaluate(data, metrics = self.metrics[0])[0]
    
    def fit(self, data):
        
        """
        For the sake of scikit-learn
        TODO: Delete if useless.
        
        Fitting function needed by scikit-learn for parameter search
        
        Parameters
        ----------
        
        data : dict of pandas.DataFrame, dict of numpy.array
            Training data including both training proper and validation
        """
        
        return self.optimize(data)
    
    def get_params(self, deep = True):
        
        """
        For the sake of scikit-learn
        TODO: Delete if useless.
        """
        
        # Suppose this estimator has parameters "alpha" and "recursive"
        return self.params
    
    def set_params(self, **parameters):
        
        """
        For the sake of scikit-learn.
        TODO: Delete if useless.
        """
        
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        
        return self    

# %% Support Vector Classifier
    
class SVC(classifier):
    
    def __init__(self, params_mesh = None, targets = None, metrics = None, 
                 features = None, data_files = None, report_files = None, 
                 model_files = None, num_epochs = None, num_runs = None, 
                 record_freq = None, batch_size = None, verbose = None, 
                 params = None):
        
        super().__init__(params_mesh, targets, metrics, features, data_files, 
                report_files, model_files, num_epochs, num_runs, record_freq, 
                batch_size, verbose, params)
        
        self.define()
        
    def define(self):
                
        from sklearn.svm import SVC
        
        clf = SVC()
        
        if self.metrics is None:
            self.metrics = ['score']
        
        self.clf = clf
        
        return clf

    def optimize(self, data_object):
        
        self.clf.fit(data_object.data['features'], data_object.data['targets'])
    
    def predict(self, data_object):
        
        return self.clf.predict(data_object.data['features'])
    
    def train(self, data):
    
        self.optimize(data)
        
        metrics = {'score': self.evaluate(data)}
        
        return metrics
    
    def test(self, data):
        
        metrics = {'score': self.evaluate(data)}
        
        return metrics    
    
    def evaluate(self, data):
        
        return self.clf.score(data['features'], data['targets'])
    
    def select(self, data):
    
        from sklearn.model_selection import GridSearchCV
        
        self.define()

        clf_select = GridSearchCV(self.clf, self.params_mesh)
        
        clf_select.fit(data['features'], data['targets'])
        
        print('clf_select.best_params_:', clf_select.best_params_)

        return clf_select
        
# %% Recurrent Neural Network
    
class RNN(classifier):
    
    """
    Variable-length Recurrent Neural Network which maps a variable-length 
    sequence to a fixed-length vector.
    
    TODO:
        
    - Figure out how we can use a dynamic batch size.
    """
    
    def __init__(self, params_mesh = None, targets = None, metrics = None, 
                 features = None, data_files = None, report_files = None, 
                 model_files = None, num_epochs = None, num_runs = None, 
                 record_freq = None, batch_size = None, verbose = None, 
                 params = None):
        
        super().__init__(params_mesh, targets, metrics, features, data_files, 
                report_files, model_files, num_epochs, num_runs, record_freq, 
                batch_size, verbose, params)
        
    def define(self, data_object, input_dim = None, target_len = None, 
               max_seq_len = None, num_hidden = None, 
               batch_size = None, cell_type = None, 
               tf_collection = 'tf_collection'):
        
        """
        Parameters
        ----------

        data_object : ``data_wrangler()`` object whose property ``data`` is a
             dict of numpy.array. This is a dictionary which contains the data, 
            both of ``features``, ``targets``, and ``seq_len``. I.e.,
            
            - ``data['features']`` are the features,
            - ``data['targets']`` are the targets, and
            - ``data['seq_len']`` are the sequence lengths of the features.
        input_dim : int
            Dimension of the input layers
        target_len : int
            Dimension of the output layer
        max_seq_len : int
            Maximum length of the input sequence
        num_hidden : int
            Number of hidden units in the RNN cell
        batch_size : int
            Fixed size of the batches
        cell_type : str
            Cell type of the RNN
        tf_collection : str
            Name of the TensorFlow collection
            
        Returns
        -------
        
        graph : TensorFlow graph
            TensorFlow graph used as default
        """
        
        data = data_object.data # NEW
        
        # Specify the default arguments and check their consistency.

        if cell_type is None: 
            cell_type = self.params['cell_type']
            
        if batch_size is None and self.batch_size is not None:
            batch_size = self.batch_size
        elif batch_size is None and self.batch_size is None:
            batch_size = data['features'].shape[0]
            self.batch_size = batch_size
        
        if num_hidden is None: 
            num_hidden = self.params['num_hidden']
        
        if max_seq_len is None: 
            max_seq_len = data['features'].shape[1]
        else:
            assert max_seq_len >= data['features'].shape[1]
            
        if target_len is None:
            target_len = data['targets'].shape[1]
        else:
            assert target_len == data['targets'].shape[1]
        
        if input_dim is None:
            input_dim = data['features'].shape[2]

        # Define the graph

        import tensorflow as tf
        
        graph = tf.Graph()
        
        with graph.as_default():
            
            # For CPU, replace ``if True:`` with ``with tf.device('/cpu:0'):`` 
            # TODO: Implement this with a ``self`` variable that toggles 
            #       between CPU and GPU implementations?
            if True: 

                tf_features = tf.placeholder(
                        tf.float32, [batch_size, max_seq_len, input_dim], 
                        name = 'tf_features')

                tf_targets = tf.placeholder(
                        tf.float32, [batch_size, target_len],
                        name = 'tf_targets')

                tf_seq_len = tf.placeholder(
                        tf.int32, [batch_size],
                        name = 'tf_seq_len')
        
                # TODO: Add the capability to load customized cells here.
        
                tf_cell = 'tf.contrib.rnn.'+cell_type+'('+str(num_hidden)+')'
                tf_cell = eval(tf_cell)
            
                tf_output, tf_state = tf.nn.dynamic_rnn(
                        tf_cell, tf_features, sequence_length = tf_seq_len, 
                        dtype = tf.float32)

                # If there are internal gates to retrieve, like the activation 
                # gates of the GRU, appended them to ``tf_output``. In such a 
                # case, sperate the actual ``tf_output`` from these gates.
                tf_gates = tf.add(
                        tf.zeros_like(tf_output[:, :, num_hidden:]), 
                        tf_output[:, :, num_hidden:], name = 'tf_gates')                
                tf_output = tf_output[:, :, :num_hidden]
                       
                # Store the last output.
                tf_last = tf.gather_nd(
                        tf_output, 
                        tf.stack([tf.range(batch_size), tf_seq_len-1], axis = 1), # batch_size
                        name = 'tf_last')
                        
                # What is the difference between the two statements below??
#                tf_weights = tf.Variable(
#                        tf.truncated_normal([num_hidden, int(tf_targets.get_shape()[1])]), 
#                        name = 'tf_weights')
                tf_weights = tf.get_variable(
                        'tf_weights', 
                        shape = [num_hidden, int(tf_targets.get_shape()[1])], 
                        initializer = tf.contrib.layers.xavier_initializer())        
            
                # Shouldn't the bias also be initialized to a random number, 
                # just like that is the case for the weights??
                tf_bias = tf.Variable(
                        tf.constant(0.1, shape = [tf_targets.get_shape()[1]]), 
                        name = 'tf_bias') 

                tf_prediction = tf.nn.softmax(
                        tf.matmul(tf_last, tf_weights) + tf_bias, 
                        name = 'tf_prediction')

                tf_cross_entropy = tf.reduce_sum(
                        -tf_targets*tf.log(tf.clip_by_value(tf_prediction, 1e-10, 1.0))/self.batch_size, 
                        name = 'tf_cross_entropy')
                
                tf_mistakes = tf.not_equal(tf.argmax(tf_targets, 1), 
                                           tf.argmax(tf_prediction, 1), 
                                           name = 'tf_mistakes')
        
                tf_error = tf.reduce_mean(tf.cast(tf_mistakes, tf.float32), 
                                          name = 'tf_error')

                # Start baseline evaluation ###################################
                
                tf_BL_prediction = tf.placeholder(
                        tf.float32, [batch_size, target_len],
                        name = 'tf_BL_prediction')
                
                tf_BL_prediction_softmax = tf.nn.softmax(
                        tf_BL_prediction, 
                        name = 'tf_BL_prediction_softmax')                

                tf_BL_cross_entropy = tf.reduce_sum(
                        -tf_targets*tf.log(tf.clip_by_value(tf_BL_prediction_softmax, 1e-10, 1.0)), 
                        name = 'tf_BL_cross_entropy')
                
                tf_BL_mistakes = tf.not_equal(tf.argmax(tf_targets, 1), 
                                           tf.argmax(tf_BL_prediction_softmax, 1), 
                                           name = 'tf_BL_mistakes')
        
                tf_BL_error = tf.reduce_mean(tf.cast(tf_BL_mistakes, tf.float32), 
                                          name = 'tf_BL_error')                          
                
                # End baseline evaluation #####################################

                # TODO: The type of optimizer should really be one of the 
                #       model parameters.
                tf_optimizer = eval('tf.train.'+self.params['optimizer'])(
                        learning_rate = self.params['learning_rate'], 
                        name = 'tf_optimizer')

                if self.maximize is False:
                    tf_optimize = tf_optimizer.minimize(
                            eval('tf_'+self.metrics[0]), 
                            name = 'tf_optimize')
                else:
                    # TODO: Use a maximizer
                    pass
        
                tf_init = tf.global_variables_initializer()
                
                # Create the TensorFlow collection
                for collection_vars in [tf_features, tf_targets, tf_seq_len, 
                                        tf_last, tf_weights, tf_bias, 
                                        tf_prediction, tf_cross_entropy, 
                                        tf_mistakes, tf_error, tf_gates, 
                                        tf_init, tf_optimize, tf_BL_prediction, 
                                        tf_BL_cross_entropy, tf_BL_mistakes, 
                                        tf_BL_error]:                    
                    graph.add_to_collection(tf_collection, collection_vars)
                
                self.saver = tf.train.Saver()
        
        self.graph = graph
        
        return graph
    
    def initialize(self, tf_collection = 'tf_collection'):
        
        """
        Initialize the TensorFlow model
        
        Parameters
        ----------
        
        tf_collection : TensorFlow collection
            TensorFlow collection
        """

        import tensorflow as tf

        # Close any pre-existing session.
        try: self.sess.close()
        except: pass
            
        # Create a new session.
        self.sess = tf.Session(graph = self.graph)
        
        with self.sess.as_default():
                    
            # Name of the TensorFlow initializer
            # TODO: Do not hardcode. Why is this not ``tf_init``??
            tf_var = 'init'
            
            # Extract the initializer from the TensorFlow collection.
            exec(tf_var+' = '+'self.graph.get_collection(\''+tf_collection+'\', \''+tf_var+'\')[0]')
            self.sess.run(eval(tf_var))
    
    def get_tf_collection(self, tf_collection = 'tf_collection'):
    
        """
        Return the names of the elements in the TensorFlow collection
        
        Parameters
        ----------
        
        tf_collection : str
            Name of the TensorFlow collection
            
        Returns
        -------
        
        A list of the names of the elements in the TensorFlow collection
        """
        
        with self.sess.as_default():
        
            return [tf_var.name for tf_var 
                    in self.graph.get_collection(tf_collection)]

    def evaluate(self, data_object, metrics = None, 
                 tf_collection = 'tf_collection', num_batches = None):
        
        """
        Evaluate TensorFlow tensors, including metrics for the model.
        
        TODO:

        - Distinguish between the reduced (e.g., averagings) and mapped (e.g., 
          predictions) Tensors.
        - Implement an automatic batch_size, or at least, suggest an optimal 
          batch size.
        
        Parameters
        ----------
        
        data_object : Object of class ``data_wrangler()``
            Data to be evaluated
        metrics : list of str, None
            List of TensorFlow tensors to be fetched. 
        num_batches : int, None
            Number of batches to evaluate. If ``None``, the number of bacthes
            is the integer number of batches that can fit into the data in 
            ``data_object.data``.
            
        Returns
        -------
        
        evaluated_tf_vars : list
            List of evaluated TensorFlow tensors representing the metrics.
        """

        from numpy import zeros, mean, nan

        # Retrieve the TensorFlow variables, i.e., the metrics.
        # IMPORTANT: The metrics should be listed in the same order as they 
        # appear in ``self.metrics``. This is because the reports that are
        # produced from the training associate the metrics and their 
        # evaluations based on this assumed order.
        if metrics is None:
            metrics = ['tf_'+metric for metric in self.metrics]
        elif type(metrics) is str:
            metrics = ['tf_'+metrics]
        elif type(metrics) in [list, tuple, set]:
            metrics = ['tf_'+metric for metric in metrics]
            
        for tf_var in metrics:    
            exec(tf_var+' = '+'self.graph.get_collection(\''+tf_collection+'\', \''+tf_var+'\')[0]')            

        assertion_msg = '\n  - self.batch_size: '+str(self.batch_size)
        assertion_msg += '\n  - data_object.nrows: '+str(data_object.nrows)
        assertion_msg += '\n  - data_object.bunch_size: '+str(data_object.bunch_size)
        assert 0 < self.batch_size <= data_object.bunch_size <= data_object.nrows, assertion_msg
        
        if num_batches is None:
            num_batches = int(data_object.nrows/self.batch_size)

        # Save the current settings of the data object so as to restore them
        # once we are done with the batch loops.
        bunch_start = data_object.bunch_start
        nrows = data_object.nrows
        bunch_size = data_object.bunch_size
            
        consumed_bunches = 0 # Number of bunches processed so far

        with self.sess.as_default():
            
            # This array contains the evaluations of the TensorFlow tensors at
            # each batch. It has to be made up object data types since the 
            # tensors may have different dimensions.
            ev_tf_vars = zeros((len(metrics), num_batches), dtype = 'O')
            
            # For each batch...
            for b in range(num_batches):

                batch_start = b*self.batch_size - consumed_bunches*data_object.bunch_size
                batch_end = (b+1)*self.batch_size - consumed_bunches*data_object.bunch_size
                
                if self.verbose >= 6:
                    print('    ............... BATCH ', b, '/',                           
                          num_batches - 1, ' [', batch_start, ', ', batch_end, 
                          '], nrows = ', data_object.nrows, sep = '')
                    try: 
                        print('    First (batch/bunch):', 
                              data_object.data_raw.iloc[batch_start].name,
                              data_object.data_raw.iloc[0].name)
                        print('    Last (batch/bunch):', data_object.data_raw.iloc[batch_end-1].name,
                              data_object.data_raw.iloc[-1].name)
                        print('    len(data_object.data_raw):', len(data_object.data_raw))
                    except: pass
                
                # TODO: Find a better way to retrieve the collection variables
                # without hardcoding their indices such as ":0"?? What are 
                # these indices for anyways?
                feed_dict = {'tf_features:0': data_object.data['features'][batch_start:batch_end], 
                             'tf_seq_len:0': data_object.data['seq_len'][batch_start:batch_end]}
                
                # Only add the targets if they exist. They will not exist for 
                # prediction tasks for example.
                try: feed_dict['tf_targets:0'] = data_object.data['targets'][batch_start:batch_end]
                except: pass
                
                # Same thing for the baseline predictions
                try: feed_dict['tf_BL_prediction:0'] = data_object.data['BL_prediction'][batch_start:batch_end]
                except: pass            

                # Evaluate each of the tensors
                for v, tf_var in enumerate(metrics):
                    
                    evaluated_tf_var = self.sess.run(eval(tf_var), feed_dict)

                    if evaluated_tf_var is None:
                        evaluated_tf_var = nan

                    # Tensor ``v`` at evaluated at batch ``b``
                    ev_tf_vars[v, b] = evaluated_tf_var
                    
                    if self.verbose >= 6:
                        print('    - Evaluated ', tf_var, ' to be ', 
                              evaluated_tf_var, sep = '')
                    
                # If we have consumed an integer number of bunches, move on to 
                # the next bunch.
                if (b+1)*self.batch_size % data_object.bunch_size == 0 and b < num_batches-1:
                    data_object.process_bunch()
                    consumed_bunches += 1

        # Once we have consumed the last batch, rewind the data object to the
        # its initial settings.
        data_object.cue(bunch_start, nrows, bunch_size)
                   
        # Average the tensors over all the batches.
        # TODO: This averaging should not be done for mapped tensors, e.g., for
        # predictions. Such mapped tensors should instead be vertically stacked
        # together.        
        
        def stitch_or_reduce(ev_tf_vars, v):

            """
            Depending on on the name of the vth metric, decide whether it 
            should be evaluated as a scalar, i.e., a mean over the batches, or
            as a stack over all the batches.
            
            Parameters
            ----------
            
            ev_tf_vars : numpy.array
                Array of the metrics for each batch
            v : int
                Index of the vth metric
            
            Returns
            -------
            
            eval_tf_var : numpy.array, float
                Evaluation of the vth metric over all the batches
            """
            
            from numpy import vstack
            eval_tf_var = vstack((ev_tf_vars[v, b] for b in range(num_batches)))
            
            # TODO: List all the possible values that amount to scalars, e.g., 
            #       cosine distance, Bhattacharyya distance, etc.
            if metrics[v][3:] in ['error', 'cross_entropy', 'BL_error', 
                      'BL_cross_entropy']: 
                eval_tf_var = mean(eval_tf_var)

            return eval_tf_var
                
        evaluated_tf_vars = [stitch_or_reduce(ev_tf_vars, v) for v 
                             in range(len(metrics))]
        
        # TODO: This was the old way of evaluating the metrics, batch by batch.
        #       Delete this once confident that the above works as desired.
#        evaluated_tf_vars = [mean(ev_tf_vars[v, :]) for v 
#                             in range(len(metrics))]
        
        return evaluated_tf_vars

    def save(self, save_as = None, prefix = 'model_tf'):
        
        """
        Save the model.
        
        TODO:
        
        - Finish here or inherit.
        
        Parameters
        ----------
        
        save_as : bool, str
            If ``True``, save the model using its metadata tag. If passed as a 
            string instead, use that string as the file name.
        """

        self.model_files['dir'] =  './model/'+self.identifier(save_as)+'/'
        
        save_tf_as = self.model_files['dir'] + self.identifier(
                '', prefix = prefix, suffix = '.ckpt')
        
        with self.sess.as_default():

            save_tf_as = self.saver.save(self.sess, save_tf_as)
            if self.verbose > 0:
                print('Model saved in file: %s' % save_tf_as)             

        # Save the object, but remove the TensorFlow objects which cannot be 
        # pickled such as sessions and graphs.
        import copy
        temp_self = copy.copy(self) # TODO: Would `self.copy()` work as well?
        del temp_self.sess, temp_self.graph, temp_self.saver
        
        super().save(temp_self, save_as)
    
    def restore(self, data_object, saved_as = None, prefix = 'model_tf'):
        
        """
        Restore the model.

        TODO:
        
        - Finish here or inherit.        
        """
        
        self.define(data_object)
        self.initialize()
        
        super().restore(saved_as)
        
        saved_tf_as = self.model_files['dir'] + self.identifier(
                '', prefix = prefix, suffix = '.ckpt')
        
        with self.sess.as_default():
            self.saver.restore(self.sess, saved_tf_as)