# -*- coding: utf-8 -*-
"""
Usefull functions for data and classifier's manipulations
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import kernels
import SVM


######################### Load raw dataset #########################

def get_train(label):
    """
    Get the train data of the corresponding group
    """
    X = pd.read_csv('../data/Xtr'+str(label)+'.csv')
    y = pd.read_csv('../data/Ytr'+str(label)+'.csv')
    X = np.array(X['seq'])
    y = np.array(([val[1] for val in y.values.tolist()]))
    return X, y


def get_test(label):
    """
    Get the test data of the corresponding group
    """
    X = pd.read_csv('../data/Xte'+str(label)+'.csv')
    X = np.array(X['seq'])
    return X



######################### Split train / test #########################

def train_test_split(X, y, train_size):
    """
    Split the data into train part and test part

    Parameters:
        - X: observations
        - y: labels
        - p: proportion of the train data
    """
    random.seed(42)

    if train_size <= 0 or train_size >= 1:
        raise ValueError('train_size is a float in (0, 1)')

    idx_0 = list(np.where(y == 0)[0])
    idx_1 = list(np.where(y == 1)[0])
    n0, n1 = len(idx_0), len(idx_1)

    idx_tr0 = random.sample(idx_0, int(train_size * n0)+1)
    idx_tr1 = random.sample(idx_1, int(train_size * n1)+1)
    idx_te0 = list(set(idx_0) - set(idx_tr0))
    idx_te1 = list(set(idx_1) - set(idx_tr1))

    idx_tr = np.concatenate((idx_tr0, idx_tr1))
    idx_te = np.concatenate((idx_te0, idx_te1))

    return X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]



######################### Grid search #########################

def accuracy(ypred, y):
    """
    Compute the accuracy between ypred and y
    """
    return np.mean(ypred==y)


def grid_search(label, X, y, svm_params, methods, train_size=0.75, graph=False):
    """
    Implementation of the cross validation

    Parameters:
        - kernel: function, kernel function
        - label: int (0, 1 or 2), label of the set of data
        - X: array, observations
        - y: array, labels
        - svm_params: array, parameters of the SVM classifier
        - kernel_params: array, parameters of the kernel function
        - train_size: float (between 0 and 1), proportion of data for the train part
        - graph: bool, plot the evolution of the accuracy wrt log(svm_params) or not

    Returns the best SVM classifier
    """
    Xtr, ytr, Xte, yte = train_test_split(X, y, train_size)

    best_score = 0
    best_clf = None

    for method in methods:
        kernel, kernel_param = kernels.select_method(method)
        print()
        scores = []

        for c in svm_params:
            print('Parameters : ' + str([method, c]))
            gram_file = "../gram_matrix/gramMat_" +  str(label) + "_" + method + ".p"
            clf = SVM.SupportVectorMachine(kernel=kernel, C=c, kernel_params=kernel_param)
            clf.fit(Xtr, ytr, gram_file)
            score = accuracy(clf.predict(Xte), yte)
            if score > best_score:
                best_score = score
                best_clf = clf
            print ("Accuracy score = " + str(score) + '\n')
            scores.append(score)

        if graph:
            plt.plot(np.log10(svm_params), scores, label='kernel_param = ' + str(kernel_param))

    if graph:
        plt.title('Evolution of the accuracy wrt log(C)')
        plt.legend()
        plt.savefig('../res/cross_val'+str(label)+'.png')
        plt.show()

    return best_clf



######################### Creation of the models #########################

def create_models(labels, params, train_size=0.75):
    """
    Create the model for each label and save the prediction on the test set
    on a csv file

    Parameters:
        - labels: list, list of the labels
        - params: dict, dictionnary containing the kernel method and
                        the SVM parameter
        - train_size: float, proportion of the train part in the data set
    """
    ytes = np.array([])

    for label in labels:
        print("\n*******Treating group " + str(label) + "*******\n")

        Xtr, ytr = get_train(label)
        Xte = get_test(label)

        Xtr, ytr, Xv, yv = train_test_split(Xtr, ytr, train_size)

        method, c = params[label]

        kernel, kernel_param = kernels.select_method(method)

        gram_file = "../gram_matrix/gramMat_" +  str(label) + "_" + method + ".p"
        clf = SVM.SupportVectorMachine(kernel=kernel, C=c, kernel_params=kernel_param)
        clf.fit(Xtr, ytr, gram_file)

        score = accuracy(clf.predict(Xv), yv)
        print ("Accuracy score = " + str(score) + '\n')

        ytes = np.concatenate((ytes, clf.predict(Xte)), axis=None)

    results = pd.DataFrame({'Id':list(range(3000)), 'Bound':ytes})
    results['Bound'] = [int((val+1)/2) for val in results['Bound']]
    results.to_csv('../predictions/Yte.csv', index=False)
