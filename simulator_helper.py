# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:45:31 2020

@author: amine
"""

import numpy as np
import pandas as pd

num_entity_types = 7
num_features = (2, 10)
num_examples = (10, 1000)
data = dict()

for k in range(num_entity_types):
    data[k] = pd.DataFrame(
        np.random.rand(
            np.random.randint(min(num_examples), max(num_examples)),
            np.random.randint(min(num_features), max(num_features))))


class Map:

    def __init__(self):

        pass

    def update(self):

        pass


print(data[3].head())
