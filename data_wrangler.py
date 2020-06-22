#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:31:35 2020

@author: Amine Laghaout
"""

import numpy as np
import pandas as pd
import tensorflow as tf


class DataWrangler:

    def __init__(self, **kwargs):
        """  This is the default data/environment wrangler. """

        # Load (or generate) the data.
        self.load()

        # Wrangle the data
        self.wrangle()

    def load(self):
        """ Load the data. """

        pass

    def wrangle(self):
        """ Wrangle the data. """

        pass

    def view(self, dataset=None, batch_num=0, num_batches=5,
             return_list=False):
        """
        View the dataset.

        Parameters
        ----------
        dataset: None, str, tf.data.Dataset
            Dataset to view. If ``None`` use the default dataset
            ``self.dataset``, if a string, use one of the four possible
            splits ``self.dataset[{'train', 'test', 'validate', 'serve'}]``.
            Otherwise, use the dataset explicitely passed as ``dataset``.
        num_batches: int
            Number of batches to view
        batch_num: int, None
            Index of the batch to display. If ``None``, return all batches.
        return_list: bool
            Return the list of ``num_batch`` batches?

        Return
        ------
        batches: list
            A list of of ``pandas.DataFrame`` corresponding to the
            ``num_batches`` retrieved.
        """

        # If no list of batches is to be returned, don't bother loading more
        # batches than is necessary to return the ``batch_num``-th batch.
        if return_list is False and batch_num is not None:
            num_batches = batch_num + 1

        batches = []

        # View the default dataset.
        if dataset is None:
            dataset = self.dataset

        # View one of the four splits of the dataset.
        elif dataset in {'train', 'test', 'validate', 'serve'}:
            dataset = self.dataset[dataset]

        # If the dataset is a tuple...
        if isinstance(dataset.element_spec, tuple):

            # ... of two elements, then assume that we are dealing with
            # supervised learning.
            if len(dataset.element_spec) == 2:

                # For each batch,
                for batch, label in dataset.take(num_batches):

                    # store the current batch as a pandas data frame.
                    batches += [assemble_dataframe(batch, label)]

        # Once ``num_batches`` are retrieved, either print every one of them,
        if batch_num is None:
            for batch_num, batch in enumerate(batches):
                print(f'Batch {batch_num}:\n', batches[batch_num])

        # or only print the one that is specified at position ``batch_num``.
        else:
            print(f'Batch {batch_num}:\n', batches[batch_num])

        if return_list:
            return batches

    def explore(self):
        """ Explore the data. """

        pass

    def split(self, split_sizes):
        """
        Split the dataset.

        Parameter
        ---------
        split_sizes: dict of int
            Dictionary that specifies the number of examples to be allocated to
            training, testing, and validation.
        """

        dataset = dict()
        dataset['train'] = self.dataset.take(split_sizes['train'])
        dataset['test'] = self.dataset.skip(split_sizes['train'])
        dataset['validate'] = dataset['test'].skip(split_sizes['test'])
        dataset['test'] = dataset['test'].take(split_sizes['train'])

        self.split_sizes = split_sizes
        self.dataset = dataset


class FromFile(DataWrangler):

    def __init__(
            self, file_path, label_name, batch_size=5, num_epochs=1,
            na_value='?', ignore_errors=True, shuffle=False, **kwargs):
        """ Generic data wrangler based on CSV files """

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.file_path = file_path
        self.label_name = label_name
        self.na_value = na_value
        self.ignore_errors = ignore_errors
        self.shuffle = shuffle

        super().__init__()

    def load(self, **kwargs):

        self.dataset = tf.data.experimental.make_csv_dataset(
            self.file_path,
            batch_size=self.batch_size,
            label_name=self.label_name,
            na_value=self.na_value,
            num_epochs=self.num_epochs,
            ignore_errors=self.ignore_errors,
            shuffle=self.shuffle,
            **kwargs)



class RotationMatrix(DataWrangler):

    def __init__(
            self, theta, num_examples, dim=2, batch_size=5):
        """
        This dataset consists of vectors of dimension ``dim`` as inputs and the
        rotated vectors by an angle ``theta`` as outputs.

        Parameters
        ----------
        theta: float
            Angle of rotation
        num_examples: int
            Number of examples
        dim: int
            Dimension of the vectors
        batch_size:
            Batch size
        """

        assert dim == 2  # Only allow 2-D vectors for now.

        self.batch_size = batch_size
        self.theta = theta
        self.dim = dim
        self.num_examples = num_examples

        super().__init__()

    def load(self):

        # Rotation matrix
        self.matrix = np.array(
            [[np.cos(self.theta), -np.sin(self.theta)],
             [np.sin(self.theta), np.cos(self.theta)]])

        # Input vector
        x = np.random.rand(self.num_examples, self.dim)

        # Output vector resulting from the multiplication
        y = np.apply_along_axis(self.rotate, 1, x)

        self.dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)).batch(self.batch_size)

    def rotate(self, x, R=None):
        """
        Parameters
        ----------
        x: numpy.array
            Input vector
        R: numpy.array, None
            Rotation matrix

        Return
        ------
        rotated_x: numpy.array
            Rotated vector
        """

        if R is None:
            R = self.matrix

        rotated_x = np.matmul(R, x)

        return rotated_x

# %% Utilities


def assemble_dataframe(batch, label):
    """
    Parameters
    ----------
    batch: tf.Tensor
        Features tensor
    label: tf.Tensor
        Labels tensor

    Return
    ------
    batch: pandas.DataFrame
        Data frame which assembles the batch with its labels into one

    TODO: The try~except block is supposed to accommodate the fact that the
    elements_spec is different depending on how the data was generated. Find a
    less hacky way to do this.
    """

    # Use this for tf.Tensors
    try:

        batch = pd.DataFrame(batch.numpy())
        label = pd.DataFrame(label.numpy())
        batch = pd.concat(
            [batch, label], axis=1, sort=False)

    # Use this for make_csv_dataset()
    except Exception:

        batch = pd.DataFrame(batch)
        label = pd.DataFrame(
            {'label': label}, index=range(len(label)))
        batch = pd.concat(
            [batch, label], axis=1, sort=False)

    batch = pd.DataFrame(batch)

    return batch


class PackNumericFeatures(object):
    """
    https://www.tensorflow.org/tutorials/load_data/csv#data_preprocessing
    """

    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for
                            feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels
