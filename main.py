#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:48:58 2018

@author: ala

TODO:
- Update file dates
"""

from matplotlib import use
import warnings

warnings.filterwarnings('ignore')
use('agg')

problem = 'SPX'

if problem is None:
    problem = input('Problem acronym? ')


