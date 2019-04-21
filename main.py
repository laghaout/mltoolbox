#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:48:58 2018

@author: Amine Laghaout

TODO:
- Update file dates
"""

from matplotlib import use
import warnings

#warnings.filterwarnings('ignore')
use('agg')

problem = 'SPX'

if problem is None:
    problem = input('Problem acronym? ')

if problem == 'digits':
    
    from problems import Digits
    
    problem = Digits(train=True, test=True)
    problem.run()
    
elif problem == 'synthetic':

    from problems import SyntheticClasses
    
    problem = SyntheticClasses()
    problem.run(train=True)
    
