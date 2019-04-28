#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:48:58 2018

@author: Amine Laghaout

TODO:
- Update file dates
"""

from matplotlib import use

use('agg')


def main(problem=None, ignore_warnings=False):

    if ignore_warnings:

        import warnings
        warnings.filterwarnings('ignore')

    if problem is None:

        from utilities import version_table

        version_table()

    elif problem == 'digits':

        from problems import Digits

        problem = Digits(train=True, test=True)
        problem.run()

    elif problem == 'synthetic':

        from problems import SyntheticClasses

        problem = SyntheticClasses()
        problem.run(train=True)


if __name__ == "__main__":
    main()
