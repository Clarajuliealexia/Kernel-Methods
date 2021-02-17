# -*- coding: utf-8 -*-
"""
Manual implementation of support vector machine classifier
"""

import pickle
import numpy as np
from cvxopt import matrix as cvxMat
from cvxopt.solvers import qp, options
from tqdm import tqdm



# Options of the quadratic solver
options['show_progress'] = False
options['maxiters'] = 250
options['abstol'] = 1e-8
options['reltol'] = 1e-6
options['feastol'] = 1e-8


class SupportVectorMachine():
    """
    Support vector machine binary classifier


    The SVM object has these main methods:
      - fit(X, y): to train the classifier on training data (X, y)
      - predict(X): predict the labels for each point in the dataset X
      - project(X): projects the points onto the hyperplane

    The SVM object has these main attributes:
      - C, kernel_params: parameters of the models
      - kernel: function of the kernel

    When it has been trained, it has:
      - b = intercept_: bias
      - a: Lagrange multipliers
      - n_support_: number of support vector
      - sv: support vectors
      - sv_y: labels of the support vectors
    """

    def __init__(self, kernel, kernel_params, C=1.0, threshold=1e-3):
        """
        Initialisation of the SVM

        Parameters:
            - kernel: function, kernel function
            - kernel_params: float or array, parameters of the kernel function
            - C: float, default 1, regularization parameter
            - threshold: float, default 1e-3, only keep the support vector of
                Lagrange multiplier > threshold
        """
        self.c = C
        self.threshold = threshold
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.convert = False
        self.a = None
        self.sv = None
        self.sv_y = None
        self.b = None


    def fit(self, X, y, gram_file=None):
        """
        Training of the SVM with the data (X, y)

        Parameters:
            - X: array, observations
            - y: array, labels
        Return:
            None
        """

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

        a = self.solve_qp(X, y, K)

        sv = a > self.threshold
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        b = 0
        for n in range(len(self.a)):
            b += self.sv_y[n]
            b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        if len(self.a) > 0:
            b /= len(self.a)
        self.b = b


    def solve_qp(self, X, y, K):
        """
        Resolution of the quadratic program

        Parameters:
            - X: array, observations
            - y: array, labels
            - K: matrix, Gram matrix
        Return:
            - a: array, result of the quadratic problem
        """
        n_samples = X.shape[0]
        P = cvxMat(np.outer(y, y) * K)
        q = cvxMat(np.ones(n_samples) * -1)

        if (self.c is None) or (self.c <= 0):
            G = cvxMat(np.diag(np.ones(n_samples) * (-1)))
            h = cvxMat(np.zeros(n_samples))
        else:
            G_top = np.diag(np.ones(n_samples) * (-1))
            h_left = np.zeros(n_samples)
            G_bot = np.identity(n_samples)
            h_right = np.ones(n_samples) * self.c
            G = cvxMat(np.vstack((G_top, G_bot)))
            h = cvxMat(np.hstack((h_left, h_right)))

        A = cvxMat(y, (1, n_samples), tc='d')
        b = cvxMat(0.0)

        # solution = qp(P, q, G, h, A, b)

        # a = np.ravel(solution['x'])
        return np.ravel(qp(P, q, G, h, A, b)['x'])


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
            K[i, i] = self.kernel(X[i], X[i], self.kernel_params)

            for j in range(i+1, n_samples):
                K[i,j] = K[j,i] = self.kernel(X[i], X[j], self.kernel_params)
        return K


    def project(self, X):
        """
        Projection of X on the hyperplane

        Parameter:
            - X: array, observations
        Return:
            array, projection of X on the support vectors plane
        """
        ypred = np.zeros(len(X))
        for i in range(len(X)):
            ypred[i] = sum(a * sv_y * self.kernel(X[i], sv, self.kernel_params)
                           for a, sv, sv_y in zip(self.a, self.sv, self.sv_y))
        return ypred + self.b


    def predict(self, X):
        """
        Prediction of the labels of X

        Parameter:
            - X: array, observations
        Return:
            - pred: array, prediction of X
        """
        pred = np.sign(self.project(X))

        if self.convert:
            pred = np.array((pred + 1 ) / 2, dtype=int)
        return pred
