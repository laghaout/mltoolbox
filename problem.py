#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:22:22 2020

@author: Amine Laghaout
"""

import os
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import visualizers as vis
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import clear_output
import data_wrangler as dw


class Problem:

    def __init__(self):

        # Wrangle the data or environment.
        self.wrangle()

        # Design the model.
        self.design()

    def wrangle(self, data=None):
        """ Wrangle the data or environment. """

        print('========== WRANGLE:')

    def explore(self, dataset=None):
        """ Explore the data. """

        print('========== EXPLORE:')

    def design(self):
        """ Design the model. """

        print('========== DESIGN:')

    def train(self, train_data=None, validate_data=None):
        """ Train the model. """

        print('========== TRAIN:')

    def train_report(self):
        """ Report on the training. """

        print('===== Train report:')

    def test(self, test_data=None):
        """ Test the model. """

        print('========== TEST:')

    def test_report(self):
        """ Report on the testing. """

        print('===== Test report:')

    def serve(self, serve_data=None):
        """ Serve the model. """

        print('========== SERVE:')

    def serve_report(self):
        """ Report on the serving. """

        print('===== Serve report:')


class Supervised(Problem):

    def __init__(self, log_dir='logs'):

        self.log_dir = os.path.join(log_dir)

        # Wrangle the features.
        self.wrangle()

        # Design the model.
        self.design()

    def explore(self, dataset=None):
        """ Explore the data. """

        print('========== EXPLORE:')

        # If the data to be explored is not specified,
        if isinstance(dataset, bool):
            # use the training data if the dataset is split,
            if isinstance(self.data.dataset, dict):
                dataset = 'train'
            # or the whole data if the dataset is not split.
            else:
                dataset = None

        # Take into account the fact that the raw data may not be available.
        if hasattr(self, 'raw_data'):
            print('Raw data:')
            self.raw_data.view(dataset)

        print('Processed data:')
        self.data.view(dataset)

    def train(self, train_data=None, validate_data=None):
        """
        Parameters
        ----------
        train_data: None, tf.data.Dataset
            Training dataset. If ``None``, use the default dataset
            ``self.data.dataset['train']``.
        validate_data: None, tf.data.Dataset
            Validation dataset. If ``None``, use the default dataset
            ``self.data.dataset['validate']``.
        """

        print('========== TRAIN:')

        if train_data is None:
            train_data = self.data.dataset['train']
        if validate_data is None:
            validate_data = self.data.dataset['validate']

        # Tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1,
            write_images=True)

        self.model.fit(
            train_data, validation_data=validate_data,
            epochs=self.epochs,
            callbacks=[tensorboard_callback],
        )

    def test(self, test_data=None):
        """
        Parameters
        ----------
        test_data: None, tf.data.Dataset
            Testing dataset. If ``None``, use the default dataset
            ``self.data.dataset['test']``.

        Return
        ------
        metrics: list
            List of metrics
        """

        print('========== TEST:')

        if test_data is None:
            test_data = self.data.dataset['test']

        metrics = self.model.evaluate(test_data)

        return metrics

    def serve(self, serve_data=None):
        """
        Parameters
        ----------
        serve_data: None, tf.data.Dataset
            Serving dataset. If ``None``, use the default dataset
            ``self.data.dataset['serve']``.

        Return
        ------
        predictions: numpy.narray
            Predictions
        """

        print('========== SERVE:')

        if serve_data is None:
            serve_data = self.data.dataset['serve']

        predictions = self.model.predict(serve_data)

        return predictions


class Unsupervised(Problem):

    pass


class Reinforcement(Problem):

    pass


class QTableDiscrete(Reinforcement):

    def __init__(
            self,
            env,
            num_episodes=10_000,        # 10_000
            max_steps_per_episode=100,  # 100
            learning_rate=0.1,          # Learning in Bellman's equation
            discount_rate=0.99,         # Discount future rewards
            exploration_rate=1,         # Exploration vs. exploitation
            max_exploration_rate=1,
            min_exploration_rate=0.01,
            exploration_decay_rate=0.001,
            q_table_displays=3):

        self.env = env
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table_displays = q_table_displays

        super().__init__()

    def wrangle(self):
        """ Define the environment. """

        print('========== WRANGLE:')

        self.env = gym.make(self.env)

    def explore(self, *kwargs):
        """ Explore the environment. """

        print('========== EXPLORE:')

        print("Dimensionality of the action space:", self.env.action_space.n)
        print("Dimensionality of the state space:",
              self.env.observation_space.n)

    def design(self):
        """
        Design the Q-table in the case of discretized action and observation
        space.
        """

        self.q_table = np.zeros(
            (self.env.observation_space.n, self.env.action_space.n))

    def train(self):

        print('========== TRAIN:')

        # Keep track of the rewards for each episode.
        self.rewards_all_episodes = []

        for episode in range(self.num_episodes):

            # Initialize a new episode.
            state = self.env.reset()

            if episode % (self.num_episodes // self.q_table_displays) == 0:
                plt.clf()
                plt.imshow(self.q_table.T)
                plt.show()

            done = False

            rewards_current_episode = 0

            for step in range(self.max_steps_per_episode):

                # Exploration-exploitation tradeoff: Epsilon greedy strategy
                if np.random.uniform(0, 1) > self.exploration_rate:
                    action = np.argmax(self.q_table[state, :])
                else:
                    action = self.env.action_space.sample()

                # Evaluate the next step based on the action.
                new_state, reward, done, info = self.env.step(action)

                # Update the Q-table using the Bellman equation.
                self.q_table[state, action] = \
                    self.q_table[state, action] * (1 - self.learning_rate) + \
                    self.learning_rate * (
                        reward +
                        self.discount_rate * np.max(self.q_table[new_state, :]))

                # Execute the next step.
                state = new_state

                # Append the new reward
                rewards_current_episode += reward

                if done is True:
                    break

            # Decay the proportion of exploration.
            self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * \
                np.exp(-self.exploration_decay_rate * episode)

            self.rewards_all_episodes.append(rewards_current_episode)

    def train_report(self):

        print('===== Train report:')

        window = self.num_episodes // 10

        # Calculate and print the average reward per thousand episodes
        rewards_per_thousand_episodes = np.split(
            np.array(self.rewards_all_episodes), self.num_episodes / window)
        count = window

        x = []
        y = []

        for r in rewards_per_thousand_episodes:
            count += window
            x.append(count)
            y.append(sum(r / window))

        vis.plot_time_series(
            x, {f'rewards/{window}': y}, xlabel='Episode',
            ylabel=f'Rewards/{window}',
            legend=False)

        if len(self.q_table.shape) == 2:
            plt.clf()
            plt.imshow(self.q_table.T)
            plt.show()

    def serve(self):

        print('========== SERVE:')

        # Watch our agent play Frozen Lake by playing the best action
        # from each state according to the Q-table

        for episode in range(3):
            state = self.env.reset()
            done = False
            print("\n*****EPISODE ", episode + 1, "*****")
            time.sleep(1)

            for step in range(self.max_steps_per_episode):

                # Show the current state of environment on screen.
                clear_output(wait=True)
                self.env.render()
                time.sleep(0.3)

                # Choose the action with highest Q-value for current state.
                action = np.argmax(self.q_table[state, :])

                # Take a new action.
                new_state, reward, done, info = self.env.step(action)

                if done:
                    clear_output(wait=True)
                    self.env.render()
                    if reward == 1:
                        print("****You reached the goal!****")
                        time.sleep(3)
                    else:
                        print("****You fell through a hole!****")
                        time.sleep(3)
                        clear_output(wait=True)
                    break

                # Set new state
                state = new_state

        self.env.close()

# %% Specific problem domains


class MountainCar(QTableDiscrete):

    def __init__(
            self,
            env,
            num_episodes=25_000,        # 25_000
            max_steps_per_episode=100,  # 100
            learning_rate=0.1,          # Learning in Bellman's equation
            discount_rate=0.95,         # Discount future rewards
            exploration_rate=1,         # Exploration vs. exploitation
            max_exploration_rate=1,
            min_exploration_rate=0.01,
            exploration_decay_rate=0.001,
            q_table_displays=3,
            discretization={'observation': 20}):

        self.env = env
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table_displays = q_table_displays
        self.discretization = discretization

        self.wrangle()
        self.design()

    def get_discrete_state(self, state):
        """ Discretize the states to fit in the Q-table. """

        discrete_state = (state - self.env.observation_space.low) /\
            self.discrete_window_size['observation']

        return tuple(discrete_state.astype(np.int))

    def wrangle(self):
        """ Define the environment. """

        print('========== WRANGLE:')

        self.env = gym.make(self.env)

        self.discrete_size = dict()
        self.discrete_window_size = dict()

        for k in self.discretization.keys():

            self.discrete_size[k] = [self.discretization[k]] * \
                len(self.env.observation_space.high)

            self.discrete_window_size[k] = (
                self.env.observation_space.high - self.env.observation_space.low) / self.discrete_size[k]

    def explore(self, *kwargs):
        """ Explore the environment. """

        print('========== EXPLORE:')

        print("Dimensionality of the action space:", self.env.action_space.n)
        print("Dimensionality of the state space:",
              self.discrete_size['observation'])

    def design(self):
        """
        Design the Q-table in the case of discretized action and observation
        space.
        """

        self.q_table = np.random.uniform(low=-2, high=0,
                                         size=(self.discrete_size['observation'] +
                                               [self.env.action_space.n]))

    def train(self):

        print('========== TRAIN:')

        # Keep track of the rewards for each episode.
        self.rewards_all_episodes = []

        for episode in range(self.num_episodes):

            # Initialize a new episode.
            state = self.env.reset()
            state = self.get_discrete_state(state)

            if episode % (self.num_episodes // self.q_table_displays) == 0:
                #                plt.clf()
                #                plt.imshow(self.q_table.T)
                #                plt.show()
                self.env.render()

            done = False

            rewards_current_episode = 0

            for step in range(self.max_steps_per_episode):

                # Exploration-exploitation tradeoff: Epsilon greedy strategy
                if np.random.uniform(0, 1) > self.exploration_rate:
                    action = np.argmax(self.q_table[state])
                else:
                    action = self.env.action_space.sample()

                # Evaluate the next step based on the action.
                new_state, reward, done, info = self.env.step(action)
                new_state = self.get_discrete_state(new_state)

                # Update the Q-table using the Bellman equation.
                self.q_table[state, action] = \
                    self.q_table[state, action] * (1 - self.learning_rate) + \
                    self.learning_rate * (
                        reward +
                        self.discount_rate * np.max(self.q_table[new_state]))

                # Execute the next step.
                state = new_state

                # Append the new reward
                rewards_current_episode += reward

                if done is True:
                    break

            # Decay the proportion of exploration.
            self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * \
                np.exp(-self.exploration_decay_rate * episode)

            self.rewards_all_episodes.append(rewards_current_episode)


class Domain(Supervised):

    def __init__(
            self, label_name='Campaign Monitor.domain.belief.actionable',
            file_path=['/', 'home', 'ala', 'Python', 'data', 'domain',
                       'domain.csv'],
            batch_size=5):
        """
        This problem evaluates whether domains were produced algorithmically.
        """

        self.batch_size = batch_size
        self.label_name = label_name
        self.file_path = os.path.join(*file_path)

        super().__init__()

    def wrangle(self):

        self.data = dw.Domain(
            self.file_path, self.label_name, batch_size=self.batch_size)

    def train(self):
        pass

    def train_report(self):
        pass

    def test(self):
        pass

    def test_report(self):
        pass

    def serve(self):
        pass

    def serve_report(self):
        pass


class DGA(Supervised):

    def __init__(
            self, label_name='DGA',
            file_path=['/', 'home', 'ala', 'Python', 'data', 'DGA',
                       'domains.csv'],
            batch_size=5):
        """
        This problem evaluates whether domains were produced algorithmically.
        """

        self.batch_size = batch_size
        self.label_name = label_name
        self.file_path = os.path.join(*file_path)

        super().__init__()

    def wrangle(self):

        self.data = dw.DGA(
            self.file_path, self.label_name, batch_size=self.batch_size)

    def train(self):
        pass

    def train_report(self):
        pass

    def test(self):
        pass

    def test_report(self):
        pass

    def serve(self):
        pass

    def serve_report(self):
        pass


class Heart(Supervised):

    def __init__(
            self, label_name='target',
            file_path=['/', 'home', 'ala', '.keras', 'datasets', 'heart.csv'],
            numeric_features=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                              'restecg', 'thalach', 'exang', 'oldpeak',
                              'slope', 'ca'],
            categorical_features=['thal'],
            epochs=40,
            categories=dict(thal=['fixed', 'normal', 'reversible']),
            split_sizes={'train': 40, 'validate': 10, 'test': 10}):
        """ This problem predicts the indidence of heart diseases. """

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.categories = categories
        self.label_name = label_name
        self.split_sizes = split_sizes
        self.epochs = epochs
        self.file_path = os.path.join(*file_path)

        super().__init__()

    def wrangle(self):

        select_columns = self.numeric_features + self.categorical_features + \
            [self.label_name]

        def load():

            return dw.FromFile(self.file_path, self.label_name,
                               select_columns=select_columns)

        # TODO: Find a smart way to keep track of both the raw data and the
        # processed data, including when the data is shuffled.
        self.data = load()
        self.raw_data = load()

        # Numeric data
        self.data.dataset = self.data.dataset.map(
            dw.PackNumericFeatures(self.numeric_features))
        self.numeric_data = tf.feature_column.numeric_column(
            'numeric', shape=[len(self.numeric_features)])
        self.numeric_data = [self.numeric_data]

        # Categorical data
        self.categorical_data = []
        for feature, vocab in self.categories.items():
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=feature, vocabulary_list=vocab)
            self.categorical_data.append(
                tf.feature_column.indicator_column(cat_col))

        # Split the data.
        self.data.split(self.split_sizes)
        self.raw_data.split(self.split_sizes)

    def design(self):

        self.model = keras.Sequential([
            keras.layers.DenseFeatures(
                self.numeric_data + self.categorical_data),
            keras.layers.Dense(13, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer='adam',
            metrics=['accuracy'])

    def train_report(self):
        """ Report on the training. """

        print('===== Train report:')

        vis.plot_time_series(
            x=self.model.history.epoch,
            y_dict={x: self.model.history.history[x] for
                    x in ['accuracy', 'val_accuracy']},
            xlabel='epoch')

    def serve(self):

        pass


class RotationMatrix(Supervised):

    def __init__(
            self, batch_size=10, dim=2, num_examples=1000, theta=.42,
            split_sizes={'train': 50, 'validate': 25, 'test': 25},
            epochs=15):
        """
        This problem attempts to infer the matrix that rotates the feature
        vectors into the target vectors.
        """

        self.batch_size = batch_size
        self.dim = dim
        self.num_examples = num_examples
        self.theta = theta
        self.split_sizes = split_sizes
        self.epochs = epochs

        super().__init__()

    def wrangle(self):

        self.data = dw.RotationMatrix(self.theta, self.num_examples)
        self.data.split(self.split_sizes)

    def design(self):

        self.layer = MyLayer(self.dim)

        self.model = tf.keras.Sequential([
            self.layer])

        # The compile step specifies the training configuration
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.001),
            loss=tf.keras.losses.MeanAbsolutePercentageError(),
            metrics=['mse'])

    def train_report(self):
        """ Report on the training. """

        print('===== Train report:')
        vis.plot_time_series(
            x=self.model.history.epoch,
            y_dict={x: self.model.history.history[x] for
                    x in ['mse', 'val_mse']},
            xlabel='epoch')

    def test_report(self):

        print('===== Test report:')
        print('Predicted kernel matrix:')
        print(self.layer.kernel.numpy())
        print('Actual matrix:')
        print(self.data.matrix)

    def serve(self):

        pass

# %% Utilities


class MyLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], self.output_dim),
            initializer='uniform',
            trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
