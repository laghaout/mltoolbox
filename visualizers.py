#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:14:06 2017

@author: Amine Laghaout
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(
        x, y_dict, fontsize=16, markersize=3, xlabel='', ylabel='',
        loc='upper left', bbox_to_anchor=(1, 1), title=None, linewidth=3,
        xtick_frequency=10, rotation=45, save_as=None, adjust_xticks=True,
        log=(False, False), legend=True):
    """ Plot the data stored in the dictionary ``y_dict`` versus ``x``. """

    plt.figure()

    for y_key in y_dict.keys():

        plt.plot(x, y_dict[y_key], label=y_key, marker='o',
                 markersize=markersize, linewidth=linewidth)
        plt.xlabel(r'%s' % xlabel, fontsize=fontsize)
        plt.ylabel(r'%s' % ylabel, fontsize=fontsize)
        plt.setp(plt.xticks()[1], rotation=rotation)
#        ax = plt.gca()

        # Use logarithmic scale?
        if log[0]:
            plt.xscale('log')  # , nonposy = 'clip'
        if log[1]:
            plt.yscale('log')  # , nonposy = 'clip'

#        if len(x) > xtick_frequency and adjust_xticks:
#            ticks_indices = np.arange(0, len(x), int(len(x)/xtick_frequency))
#            plt.xticks(ticks_indices)
#            try:
#                ax.set_xticklabels(x[ticks_indices])
#            except Exception:
#                pass

        if legend:
            plt.legend(
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                fontsize=fontsize)

        if title is not None:
            plt.title(r'%s' % title, fontsize=fontsize)

    plt.grid()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None, save_as=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure()

    if cmap is None:
        cmap = plt.cm.Blues

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight')

    plt.show()
    plt.clf()
