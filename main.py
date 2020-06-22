#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:13:20 2020

@author: Amine Laghaout

Requirements
------------
- Duck typing
- Extraction of all the parameters to the interface, i.e. no hard-coding
- All data should be handled as a ``tf.data`` whenever possible

To-do
-----
- Defaults handling
- Config file
"""

import sys
import problem as pro


def main(
        problem='domain',
        explore=True, train=False, test=False, serve=False):

    print(f'======================================== {problem}')

    if len(sys.argv) >= 2:
        problem = sys.argv[1]
    elif len(sys.argv) >= 3:
        explore = sys.argv[2]
    elif len(sys.argv) >= 4:
        train = sys.argv[3]
    elif len(sys.argv) >= 5:
        test = sys.argv[4]
    elif len(sys.argv) >= 6:
        serve = sys.argv[5]

    # Create the object
    if problem == 'heart':
        problem = pro.Heart(file_path=['data', 'heart.csv'])
    elif problem == 'DGA':
        problem = pro.DGA(batch_size=15)
    elif problem == 'domain':
        problem = pro.Domain(batch_size=1000)
    elif problem == 'rotation_matrix':
        problem = pro.RotationMatrix(epochs=10)
    elif problem in ['FrozenLake-v0', 'FrozenLake8x8-v0']:
        problem = pro.QTableDiscrete(
            problem,
            q_table_displays=15)
    elif problem in ['MountainCar-v0']:
        problem = pro.MountainCar(problem)

        explore = True
        train = True
        test = False
        serve = False

    else:
        print(f'WARNING: There is no problem named {problem}.')
        problem = pro.Problem()

    # Explore
    if explore:
        problem.explore(explore)

    # Train
    if train:
        problem.train()
        problem.train_report()

    # Test
    if test:
        _ = problem.test()
        problem.test_report()

    # Serve
    if serve:
        _ = problem.serve()
        problem.serve_report()

    return problem


if __name__ == '__main__':
    problem = main()

#problem.data.view(num_batches=3, batch_num=None)
