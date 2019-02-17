#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:48:58 2018

@author: ala

TODO:
- Update file dates
"""

from problems import Digits

digits = Digits(nex=None, algo='MLP', encoder=True)
digits.run(wrangle=True, select=False, train=False, test=False)
digits.data.view()

hum_in = digits.data.raw.input
hum_out = digits.data.raw.output
mac_in = digits.data.input
mac_out = digits.data.output

#digits.wrangle()
#
#A = digits.data.output
#
#print(A[:10])
#print(digits.data.raw.output[:10])
#
#cutoff = 1500
##
##digits.select(digits.data.input[:cutoff], digits.data.output[:cutoff])
#digits.train(digits.data.input[:cutoff], digits.data.output[:cutoff])
#prediction = digits.test(digits.data.input[cutoff:], digits.data.output[cutoff:])

#digits.run(test=True)

#from visualizers import Plot2D
#
#digits.pipeline.steps[0][1].model.history.history
#
#epochs = range(1, len(report['train_curve']['acc'])  + 1) # self.params.epochs
#
#try:
#    Plot2D(x=epochs,
#           y=(report['train_curve']['acc'],
#               report['train_curve']['val_acc']),
#           title='Model accuracy', xlabel='Epoch', ylabel='Accuracy',
#           legend=['Train', 'Test'], marker=marker, show=show)
#except BaseException:
#    pass
#
#try:
#    Plot2D(x=epochs,
#           y=(report['train_curve']['loss'],
#               report['train_curve']['val_loss']),
#           title='Model loss', xlabel='Epoch', ylabel='Loss',
#           legend=['Train', 'Test'], marker=marker, show=show)
#except BaseException:
#    pass
