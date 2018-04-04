#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:36:24 2017

@author: Amine Laghaout
"""

import warnings
import problems as pro

warnings.filterwarnings('ignore')
verbose = 0

#%%%%%%%%%%%%%%%%%%%% DUMMY 

#problem_dummy = pro.dummy(verbose = verbose, input_dim = 20)
#results_dummy = problem_dummy.pipeline(
#        examine = True, train = True, test = True, serve = True)

#%%%%%%%%%%%%%%%%%%%% IRIS 

#problem_iris = pro.iris(verbose = verbose) 
#results_iris = problem_iris.pipeline(
#        examine = False, select = True, train = False, test = False, 
#        params = {'marker': 'x', 'key_only': True})

#%%%%%%%%%%%%%%%%%%%% DIGITS

#problem_digits = pro.digits(verbose = verbose)
#results_digits = problem_digits.pipeline(
#        examine = True, train = True, test = True)

#%%%%%%%%%%%%%%%%%%%% RANDOM WALK

#problem_random_walk = pro.random_walk(
#        verbose = verbose, input_dim = 2, nrows = 5000, max_seq_len = 200)
#results_random_walk = problem_random_walk.pipeline(
#        examine = True, train = True, test = True, serve = False)

#%%%%%%%%%%%%%%%%%%%% IMDB

#problem_imdb = pro.imdb(verbose = verbose)
#results_imdb = problem_imdb.pipeline(examine = True, train = True, test = True)
