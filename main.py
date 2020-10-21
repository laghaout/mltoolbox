#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout

TODO code:

- Create subfunctions for the numerical and categorical transformers.
+ Place everything under ``hyperparams`` and ``data_params``.
- Hyperparameter optimization
- Thorough comments
- Use JSON files for inputs (as a last step).
- Capture the stdout into a file under the learner_dir which is timestamped and
  concatenated.

TODO documentation:

- Create a table with technology and level of detail (taxonomy) as dimensions.
- Detailed UML and diagram
- Explain how the metrics are synonymous with results and they may also include
  results (e.g., predictions, timing metrics)
- Explain how the generic is separate from the detailed. I'll maintain the
  generic
- File structure
- TensorBoard
- Demos
- Go over the metrics
- Warning about Scikit-learn: Not scalable
- Provide some feedback.

TODO requirements specification:

- Data with td.data.Dataset
- tf.keras
- Exploration: Pearson correlation heatmap
- Visualization with Seaborn: https://www.tensorflow.org/tutorials/keras/regression
"""

import sys
import learners.learner as lea
import learners.utilities as util


def main(learner='learner',
         explore=True, select=True, train=True, test=True, serve=True):
    """
    This function is used to invoke a pre-defined learner object from the
    command line.

    Parameters
    ----------
    learner: Learner, str
        Learner to be instantiated.
    explore: bool
        Explore the data?
    select: bool
        Select the model?
    train: bool
        Train the model?
    test: bool
        Test the model?
    serve: bool
        Serve the model?

    Return
    ------
    learner: learner.Learner
        Learner object.
    """

    # Process the command-line arguments.
    [learner, explore, select, train, test, serve] = util.set_argv(
        [learner, explore, select, train, test, serve], sys.argv)

    # Instantiate one of the predefined learners whenever a string is provided.
    if isinstance(learner, str):
        learner = instantiate(learner)

    learner.run(explore, select, train, test, serve)

    return learner


def instantiate(learner):
    """
    Whenever a learner is referred to by its (string) name, it means that the
    it is predifined.

    TODO: Would it be more elegant to replace this by a factory method?

    Parameters
    ----------
    learner: str
        Name of the learner to instantiate.

    Return
    ------
    learner: learner.Learner
        Instantiated learner.
    """

    assert isinstance(learner, str)

    # Predict the incidence of heart diseases.
    if learner == 'heart':
        learner = lea.Heart(learner_dir=learner)

    # Predict the price of housing in Boston.
    elif learner == 'boston':
        learner = lea.Boston(learner_dir=learner)

    # Infer the rotation matrix that maps a vector to another.
    elif learner == 'rotation_matrix':
        learner = lea.RotationMatrix(learner)

    # Learn the best path over a frozen lake.
    elif learner in ['FrozenLake-v0', 'FrozenLake8x8-v0']:
        learner = lea.FrozenLake(learner)

    # Learn the best propultion strategy to climb a mountain.
    elif learner in ['MountainCar-v0']:
        learner = lea.MountainCar(learner)

    else:
        print(f'WARNING: There is no learner named ``{learner}\'\'.',
              'Running the default template instead.')
        learner = lea.LearnerChild(learner, some_argument='my_argument')

    return learner


if __name__ == '__main__':
    learner = main()
    metrics = learner.metrics
