# -*- coding: utf-8 -*-
"""
Manual implementation of kernel ridge regression classifier
"""

import pickle
import numpy as np
from tqdm import tqdm


class KernelRidge:
    """
    Kernel Ridge Regression binary classifier


    The KRR object has these main methods:
      - fit(X, y): to train the classifier on training data (X, y)
      - predict(X): predict the labels for each point in the dataset X

    The KRR object has these main attributes:
      - lambda_param, kernel_params: parameters of the models
      - kernel: function of the kernel

    When it has been trained, it has:
      - alpha = coefficients
    """

    def __init__(self, kernel, kernel_param, lambda_param):
        """
        Initialisation of the KRR

        Parameters:
            - kernel: function, kernel function
            - kernel_params: float or array, parameters of the kernel function
            - lambda_param: float, regularization parameter
        """
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.lambda_param = lambda_param
        self.Xtr = None
        self.convert = False
        self.alpha = None


    def fit(self, X, y, gram_file=None):
        """
        Training of the KRR with the data (X, y)

        Parameters:
            - X: array, observations
            - y: array, labels
        Return:
            None
        """
        self.Xtr = X

        # Verify if the labels are -1 or 1
        # if not, convert 0, 1 to -1, 1
        if set(y) == {0, 1}:
            self.convert = True
            y = np.array((y * 2) - 1, dtype=int)

        if gram_file is not None:
            try:
                K = pickle.load(open(gram_file, 'rb'))
            except FileNotFoundError:
                K = self.gram_matrix(X)
                pickle.dump(K, open(gram_file, 'wb'))
        else:
            K = self.gram_matrix(X)

        self.alpha = np.dot(np.linalg.inv(K + X.shape[0]*self.lambda_param
                                          *np.eye(X.shape[0])), y)


    def predict(self, X):
        """
        Prediction of the labels of X

        Parameter:
            - X: array, observations
        Return:
            - pred: array, prediction of X
        """
        pred = np.empty((X.shape[0]), dtype=np.float)

        for i in range(X.shape[0]):
            pred[i] = np.sign(np.sum([np.dot(self.alpha[_],
                                          self.kernel(self.Xtr[_], X[i]))
                                   for _ in range(0, self.Xtr.shape[0])]))
        if self.convert:
            pred = np.array((pred + 1 ) / 2, dtype=int)
        return pred


    def gram_matrix(self, X):
        """
        Computation of the Gram matrix

        Parameter:
            - X: array, observations
        Return:
            K: matrix, gram matrix of X
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))

        for i in tqdm(range(n_samples), desc='Building gram matrix'):
            K[i, i] = self.kernel(X[i], X[i], self.kernel_param)

            for j in range(i+1, n_samples):
                K[i,j] = K[j,i] = self.kernel(X[i], X[j], self.kernel_param)
        return K
