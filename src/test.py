# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 16:59:19 2021

@author: clara
"""

import utils


label = 0

# parameters of the classifier
SVMparams = [0.006, 0.005]
methods = ['KS_7']


Xtr, ytr = utils.get_train(label)
utils.grid_search(label, Xtr, ytr,
                  SVMparams, methods,
                  train_size=0.75, graph=False)
