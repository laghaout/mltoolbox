#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:24:45 2018

@author: Amine Laghaout
"""

def plot2D(x, y, xlabel = '', ylabel = '', title = '', fontsize = 16, 
           smooth = None, save_as = None, axis_range = None, grid = True, 
           log = (False, False), legend = '', marker = None, 
           markersize = None, xticks = None):

    """
    Generic function for plotting in two dimensions.
    Parameters
    ----------
    
    x : list
        List of the independent variables
    y : list or tuple of lists
        List or tuple of lists of the dependent variables
    xlabel : str
        Label for the independent variable
    ylabel : str
        Label for the dependent variable
    title : str
        Title for the plot
    fontsize : int
        Size of the labels
    smooth : list
        Two-element list of the parameters to the Savitzky-Golay filter
    save_as : str
        Pathname for where to save the plot
    axis_range : tuple
        Tuple of four numbers specifying the range of the plot, namely 
        ``(min(x), max(x), min(y), max(y))``
    log : tuple
        Tuple of two booleans (b1, b2) specifying whether the x-axis and y-axis
        should be logarithmic.
    legend : list of str
        Names of the curves
    xticks : None, list, tuple
        Labels of the ticks at the abcissa. If a tuple is provided, the first
        element consists of the list of tick labels and the second specifies
        their orientation ('vertical' or 'horizontal').
    
    Returns
    -------
    
    plt : plot
        The string character corresponding to the scalar type.
    """    

    import numpy as np
    import scipy.signal as sp
    import matplotlib.pyplot as plt

    plt.figure()

    # Font
    #plt.rc('font', family = fontFamily)
    
    # Plot several curves or...
    if type(y) is tuple: 
        
        if legend in ['', None]:
            legend = ['']*len(y)
        
        for y_element_index, y_element in enumerate(y):
            
            plt.plot(x, y_element, label = legend[y_element_index], 
                     marker = marker, markersize = markersize)
    
    # ... a single curve?
    else:
        
        plt.plot(x, y, legend, marker = marker, markersize = markersize)
    
        # Superimpose a smoothed line?
        if smooth != None:
            
            yhat = sp.savgol_filter(y, smooth[0], smooth[1])
            plt.plot(x, yhat, color = 'red')

    # Use logarithmic scale?    
    if log[0]:
        plt.xscale('log', nonposy = 'clip')
    if log[1]:
        plt.yscale('log', nonposy = 'clip')
    
    # Labels, legend, and title    
    plt.xlabel(r'%s' % xlabel, fontsize = fontsize)
    plt.ylabel(r'%s' % ylabel, fontsize = fontsize)
    plt.title(r'%s' % title, fontsize = fontsize)    
    plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1), 
               fontsize = fontsize)

    # Abcissa labels
    if xticks is not None:
        if type(xticks) is not tuple:
            xticks = (xticks, 'horizontal')
        plt.xticks(x, xticks[0], rotation = xticks[1])
    
    # Axis
    if axis_range == None and type(y) is not tuple:

        axis_range = (min(x), max(x), min(y), max(y))

    elif axis_range == None and type(y) is tuple:

        min_y = np.inf
        max_y = -np.inf
        
        for y_element in y:

            if min(y_element) < min_y:
                min_y = min(y_element)
            if max(y_element) > max_y:
                max_y = max(y_element) 
                
        axis_range = (min(x), max(x), min_y, max_y)
    
    plt.axis(axis_range)
    
    # Grid
    if grid:
        plt.grid()
    
    if save_as != None:
        plt.savefig(save_as, bbox_inches = 'tight')

    plt.show()
    plt.clf()    