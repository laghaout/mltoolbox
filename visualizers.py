#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:14:06 2019

@author: ala
"""

class Visualizer:

    def __init__(
            self, 
            default_args=dict(save_as=None), 
            **kwargs):

        from utilities import args_to_attributes

        args_to_attributes(self, default_args, **kwargs)

class Plot2D(Visualizer):

    from numpy import logspace

    def __init__(
            self,
            default_args=dict(
                linewidth=5, show=True, marker=None, legend=None, xlabel='', 
                ylabel='', title='', fontsize=16, smooth=None, save_as=None, 
                axis_range=None, grid=True, log=(False, False),  
                markersize=None, xticks=None, drawstyle=None, 
                fill_between=False,
                ),
            **kwargs):

        import matplotlib.pyplot as plt
        from utilities import parse_args

        kwargs = parse_args(default_args, kwargs)

        super().__init__(**kwargs)  
        
        x = self.x
        y = self.y
        
        if not self.show:
            plt.ioff()

        plt.figure()

        # Font
        #plt.rc('font', family = fontFamily)

        # Plot several curves or...
        if isinstance(y, tuple):

            if legend in ['', None]:
                legend = [''] * len(y)

            for y_element_index, y_element in enumerate(y):

                plt.plot(x, y_element, label=legend[y_element_index],
                         marker=marker, markersize=markersize,
                         drawstyle=drawstyle, linewidth=linewidth)

        # ... a single curve?
        else:

            plt.plot(x, y, label=legend, marker=marker, linewidth=linewidth,
                     markersize=markersize, drawstyle=drawstyle)

            # Superimpose a smoothed line?
            if smooth is not None:

                yhat = sp.savgol_filter(y, smooth[0], smooth[1])
                plt.plot(x, yhat, color='red')

            if fill_between:
                plt.fill_between(x, 0, y, step='mid')

        # Use logarithmic scale?
        if log[0]:
            plt.xscale('log')  # , nonposy = 'clip'
        if log[1]:
            plt.yscale('log')  # , nonposy = 'clip'

        # Labels, legend, and title
        plt.xlabel(r'%s' % xlabel, fontsize=fontsize)
        plt.ylabel(r'%s' % ylabel, fontsize=fontsize)
        plt.title(r'%s' % title, fontsize=fontsize)
        if legend is not None and legend:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1),
                       fontsize=fontsize)

        # Abcissa labels
        if xticks is not None:
            if not isinstance(xticks, tuple):
                xticks = (xticks, 'horizontal')
            plt.xticks(x, xticks[0], rotation=xticks[1])

        # Axis
        if axis_range is None and not isinstance(y, tuple):

            axis_range = (min(x), max(x), min(y), max(y))

        elif axis_range is None and isinstance(y, tuple):

            min_y = np.inf
            max_y = -np.inf

            for y_element in y:

                # print('>>> y_element:', y_element, 'y:', y)  # TODO remove

                if min(y_element) < min_y:
                    min_y = min(y_element)
                if max(y_element) > max_y:
                    max_y = max(y_element)

            axis_range = (min(x), max(x), min_y, max_y)

        plt.axis(axis_range)

        # Grid
        if grid is not False:
            if grid is True:
                plt.grid()
            else:
                plt.grid(**grid)

        if save_as is not None:
            plt.savefig(save_as, bbox_inches='tight')

        if self.show:
            plt.show()

        plt.clf()
        
def plotTimeSeries(
        x, y_dict, fontsize=16, markersize=3, xlabel='', ylabel='',
        loc='upper left', bbox_to_anchor=(1, 1), title=None, linewidth=3,
        xtick_frequency=10, rotation=45, save_as=None, adjust_xticks=True,
        log=(False, False), legend=True):

    """
    Plot the data stored in the dictionary ``y_dict`` versus ``x``.
    """

    import matplotlib.pyplot as plt
    from numpy import arange

    plt.figure()

    for y_key in y_dict.keys():

        plt.plot(x, y_dict[y_key], label=y_key,
                 markersize=markersize, linewidth=linewidth)
        plt.xlabel(r'%s' % xlabel, fontsize=fontsize)
        plt.ylabel(r'%s' % ylabel, fontsize=fontsize)
        plt.setp(plt.xticks()[1], rotation=rotation)

        # Use logarithmic scale?
        if log[0]:
            plt.xscale('log')  # , nonposy = 'clip'
        if log[1]:
            plt.yscale('log')  # , nonposy = 'clip'

        if len(x) > xtick_frequency and adjust_xticks:
            plt.xticks(arange(0, len(x), int(len(x)/xtick_frequency)))

        if legend:
            plt.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=fontsize)

        if title is not None:
            plt.title(r'%s' % title, fontsize=fontsize)

    plt.grid()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight')
        
    plt.show()

            