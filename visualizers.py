#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:36:43 2017

@author: Amine Laghaout
"""

def visualize_learning(param, trials, fontsize = 16, s = 20, linewidth = 0.01, 
                       alpha = 1, save_as = None, title = ''):
    
    """
    Visualize the Hyperopt trials for the model parameter ``param``.
    
    TODO: For discrete values, plot the y axis differently.
    
    Parameters
    ----------
    
    metric : str
        Metric whose trials to plot
    trials : hyperopt.base.Trials
        Hyperopt record of the trials
    fontsize : int
        Font size for the plot labels
    s : int
        Size of the plot markers
    linewidth : float
        Size of the data points
    alpha : float
        Alpha transparency of the data points
    save_as : str, None
        Name of the saved plot
    """
    
    import matplotlib.pyplot as plt
    
    f, ax = plt.subplots(1)
    
    xs = [t['tid'] for t in trials.trials]
    ys = [t['misc']['vals'][param][0] for t in trials.trials]
    
#    print('************')
#    print(len(xs))
#    print(len(ys))
    
    plt.axis([min(xs), max(xs), min(ys), max(ys)])
    plt.scatter(xs, ys, s = s, linewidth = linewidth, alpha = alpha)
    
    plt.xlabel('Trials', fontsize = fontsize)
    plt.ylabel(param, fontsize = fontsize)
    if title != '':
        plt.title(title, fontsize = fontsize)
    plt.grid()
    
    if save_as is not None:
        plt.savefig(save_as, bbox_inches = 'tight')

def plotParamsSpace(params_mesh, shown_metric, metrics, fontsize = 10, 
                    title = None, xlabel = '', ylabel = '', save_as = None):

    """
    This function plots the evaluations of the metric ``shown_metric`` for each
    point in the parameter space. It is used mostly for summarizing the model
    selection.
    
    Parameters
    ----------
    
    params_mesh : tuple
        Tuple of the lists of parameters making up the mesh of the parameter 
        space
    shown_metric : str
        Metric to show
    metrics : dict
        Dictionary of metrics returned by the parameters selection 
        ``select()``.
    fontsize : int
        Reference font size for the x and y ticks
    xlabel : str
        Label for the "horizontal" parameters
    ylabel : str
        Label for the "vertical" parameters in the case of a 2D parameter space
        or label of the shown metric for the case of a 1D parameter space
    title : str
        Title of the plot
    save_as : str
        Pathname for where to save the plot
    """
    
    from numpy import zeros
    import matplotlib.pyplot as plt

    # Two model parameters are considered in the mesh.
    if len(params_mesh) == 2:
            
        (rows, cols) = params_mesh    
    
        H = zeros((len(rows), len(cols)))
        
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                H[r, c] = metrics[(row, col)]['valid'][shown_metric]
            
        plt.imshow(H)
        
        plt.xticks(range(len(cols)), cols, fontsize = fontsize)  
        plt.yticks(range(len(rows)), rows, fontsize = fontsize)  
        
        plt.tick_params(axis = 'x',     # changes apply to the x-axis
                        which = 'both', # both major and minor ticks are affected
                        bottom = 'off', # ticks along the bottom edge are off
                        top = 'off',    # ticks along the top edge are off
                        labelbottom = 'on')
    
    # A single model parameter is considered in the mesh.
    elif len(params_mesh) == 1:
        
        title = ''
        ylabel = shown_metric
        
        # TODO: Implement a simple bar plot here
        pass
    
    # Labels and title
    plt.xlabel(r'%s' % xlabel, fontsize = round(1.7*fontsize))
    plt.ylabel(r'%s' % ylabel, fontsize = round(1.7*fontsize))
    if title is None:
        title = shown_metric
    plt.title(r'%s' % title, fontsize = round(1.7*fontsize))
    
    if save_as is not None:
        plt.savefig(save_as, bbox_inches = 'tight')    
    
    plt.show()

def plot2Dhist(x, xlabel = '', ylabel = '', title = '', fontsize = 16, 
           smooth = None, save_as = None, axis_range = None, grid = True, 
           log = (False, False), legend = '', marker = None, 
           markersize = None, xticks = None, norm_hist = True, bins = None,
           kde = True):

    import numpy as np
    import pandas as pd
    from scipy import stats, integrate
    import matplotlib.pyplot as plt
    
    import seaborn as sns
    sns.set(color_codes = True)
    
    plt.figure()

    for i, data_set in enumerate(x):
        sns.distplot(data_set, norm_hist = norm_hist, label = legend[i], kde = kde,
                     bins = bins)
        
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
    if axis_range is not None:
        plt.axis(axis_range)
    
    # Grid
    if grid:
        plt.grid()
    
    if save_as != None:
        plt.savefig(save_as, bbox_inches = 'tight')

    plt.show()
    plt.clf()

    
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
        
#    return plt

def progress_bar(current, total, every = None, verbose = True, 
                 decimal_digits = 2):
    
    """
    Progress bar
    
    Parameters
    ----------
    
    current : float, int
        Current count
    total : float, int
        Total number of counts
    every : float, int
        Update the progress every steps of size ``every``
    verbose : bool
        Print the progress to screen
    decimal_digits : int
        Precision
    """

    if every is None:
        every = round(total/10 - total%10)

    if verbose:
        
        percentage_complete = round(100*current/total, decimal_digits)
                    
        if current % every == 0:
        
            print(percentage_complete, '%|', sep = '', end = '')

def dashboard(message):

    """
    Starter code for a dashboard. As is, it prints the message ``message`` in
    a pop-up window.
    
    Parameters
    ----------
    
    message : str
        String to display in a pop-up window
    """
    
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    
    label = QtWidgets.QLabel(message)
    
    window.setCentralWidget(label)
    window.show()
    app.exec_()

    # Using Flask    
    """
    from flask import Flask
    app = Flask(__name__)
    
    @app.route("/")
    def hello():
        return "<html><body>ney!<b>yay!</b></body></html>"
    
    if __name__ == "__main__":
        app.run()    
    """
    
    """
    import sys
    from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit
    from PyQt5.QtGui import QIcon
     
    class App(QWidget):
     
        def __init__(self):
            super().__init__()
            self.title = 'PyQt5 input dialogs - pythonspot.com'
            self.left = 10
            self.top = 10
            self.width = 640
            self.height = 480
            self.initUI()
     
        def initUI(self):
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)
     
            self.getInteger()
            self.getText()
            self.getDouble()
            self.getChoice()
     
            self.show()
     
        def getInteger(self):
            i, okPressed = QInputDialog.getInt(self, "Get integer","Percentage:", 28, 0, 100, 1)
            if okPressed:
                print(i)
     
        def getDouble(self):
            d, okPressed = QInputDialog.getDouble(self, "Get double","Value:", 10.50, 0, 100, 10)
            if okPressed:
                print( d)
     
        def getChoice(self):
            items = ("Red","Blue","Green")
            item, okPressed = QInputDialog.getItem(self, "Get item","Color:", items, 0, False)
            if ok and item:
                print(item)
     
        def getText(self):
            text, okPressed = QInputDialog.getText(self, "Get text","Your name:", QLineEdit.Normal, "")
            if okPressed and text != '':
                print(text)
     
    if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = App()
        sys.exit(app.exec_())    
    """

