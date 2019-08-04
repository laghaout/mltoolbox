#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

import mltoolbox.utilities as util
import mltoolbox.problems as pro
import warnings
import sys


def main(problem=None, ignore_warnings=False):

    if len(sys.argv) >= 2:
        problem = sys.argv[1]
    if len(sys.argv) >= 3:
        ignore_warnings = sys.argv[2]

    if ignore_warnings:
        warnings.filterwarnings('ignore')

    # TODO: Replace this if~else by a factory method.
    if problem is None or problem == 'version_table':
        util.version_table()

    elif problem == 'BostonHousing':
        problem = pro.BostonHousing()
        problem.run(train=True, test=True)

    return problem


if __name__ == "__main__":
    main()
