#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:32:28 2018

@author: ala
"""

from sklearn.model_selection import ParameterGrid
param_grid = {'a': [1, 2], 'b': [True, False]}
list(ParameterGrid(param_grid)) == (
   [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    {'a': 2, 'b': True}, {'a': 2, 'b': False}])
A = ParameterGrid(param_grid)

#%% Show dependence of relevance on the data set for financials
if False:
    import problems as pro
    
    problem = pro.financials(
            name = 'financials MLP', algo = 'MLP', verbose = 1, 
            nrows_train = 2000, nrows_test = 2000)

    problem.data_object_test.input_names = problem.data_object_train.input_names
    
    problem1 = problem.pipeline(
            examine = problem.data_object_train)
    problem2 = problem.pipeline(
            examine = problem.data_object_test)

#%% Relevance of financial features
if False:
    
#    tickers = problem_financials_ETC.data_object_train.data_raw.columns.tolist()
#    tickers = [col.split('_') for col in tickers]
#    tickers = [ticker[0] for ticker in tickers]
#    tickers = list(set([ticker for ticker in tickers if len(ticker) in (2, 3)]))
#    print(tickers)
    
    #tickers = ['GE', 'MCD', 'VZ', 'AES', 'DOW', 'JPM', 'WMT', 'PFE', 'XOM', 'RIG']

    import problems as pro
    verbose = True
    
    for ticker in ['GE', 'MCD', 'VZ', 'AES', 'DOW', 'JPM', 'WMT', 'PFE', 'XOM', 'RIG']:
        
        problem_financials_ETC = pro.financials(regex = '*', ticker = ticker, 
                name = '*** '+ticker+' [all]', verbose = verbose)
        results_financials_ETC = problem_financials_ETC.pipeline(
                examine = True)
    
        print('+ Filtered')
        problem_financials_ETC = pro.financials(ticker = ticker, 
                name = ticker+' [filtered]', verbose = verbose)
        results_financials_ETC = problem_financials_ETC.pipeline(
                examine = True)

#%% Importance of the features
if False: 
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    forest = problem_domains_ETC.classifier_object.model
    
    X = problem_domains_ETC.data_object_train.data.input
    features = problem_domains_ETC.data_object_train.input_names
    
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.yticks(range(X.shape[1]), [features[x] for x in indices])
    plt.ylim([-1, X.shape[1]])
    plt.show()

#%% LSTM and CNN for sequence classification in the IMDB dataset
if False: 
    
    import numpy
    from keras.datasets import imdb
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.convolutional import Conv1D
    from keras.layers.convolutional import MaxPooling1D
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    
    # fix random seed for reproducibility
    
    numpy.random.seed(7)
    
    # load the dataset but only keep the top n words, zero the rest
    
    nrows = 133
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    
    (X_train, y_train), (X_test, y_test) = (X_train[:nrows], y_train[:nrows]), (X_test[:nrows], y_test[:nrows])
    (X_train_old, y_train_old), (X_test_old, y_test_old) = (X_train, y_train), (X_test, y_test)
    
    # truncate and pad input sequences
    
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    
    # create the model
    
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    