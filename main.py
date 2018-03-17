#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:36:24 2018

@author: Amine Laghaout
"""

import problems as pro

##%% Dummy
#problem = pro.dummy()
#results = problem.pipeline(
#        examine = True, train = True, test = True, serve = True)
#
##%% Domains
problem = pro.domains(verbose = 1, 
                      nrows_train = 1000, nrows_test = 1000, 
                      algo = 'MLP')
results = problem.pipeline(examine = True, train = True, test = True)
#
##%% Iris
#problem = pro.iris(verbose = 0)
#results = problem.pipeline(train = True, test = True)
#
##%% Digits
#problem = pro.digits(verbose = 0)
#results = problem.pipeline(train = True, test = True)
#
#%% Random walk
#problem = pro.random_walk(input_dim = 1, nrows = 10000, max_seq_len = 100)
#results = problem.pipeline(train = True, test = True)
#
##%% IMDB
#problem = pro.imdb(verbose = 1)
#results = problem.pipeline(train = True, test = True)
#
##%% Financials
#problem = pro.financials(verbose = 0)
#results = problem.pipeline(train = True, test = True)
