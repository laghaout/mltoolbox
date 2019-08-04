#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:27:09 2019

@author: Amine Laghaout
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

def tester():

    data = load_iris()

    data = pd.DataFrame(
        np.hstack((data.data, np.reshape(data.target, (data.target.size, 1)))),
        columns=data.feature_names+['target'])

    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    data['target'] = data['target'].map(lambda x: target_names[int(x)])

    return data

