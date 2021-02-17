# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 17:31:54 2021

@author: clara
"""

import numpy as np
cimport numpy as np
cimport cython


def ssk_cython(s, t, int k, float l):
    """
    Recursive SSK implementation.
    :param s: document #1
    :param t: document #2
    :param k: subsequence length
    :param l: weight decay (lambda)
    :return: similarity of given documents
    return:
    """

    cdef float K_st = _compute_K(s, t, k, l, _compute_K_prime(s, t, k, l))
    return K_st


def _compute_K(s, t, int k, float l, np.ndarray[np.float_t, ndim=3] K_prime):
    """
    Compute and return the K in a recursive manner using precomputed K'
    """
    cdef float K_val = 0
    cdef int m

    for m in range(len(s)+1):
        if min(len(s[:m]), len(t)) < k:
            continue

        K_val += l**2 * sum([K_prime[k-1][len(s[:m])-1][j] for j in _find_all_char_indices(s[m-1], t)])

    return K_val

@cython.boundscheck(False)
def _compute_K_prime(s, t, int k, float l):
    """
    Compute and return K' using the efficient DP algorithm (K'')
    """
    cdef int M = len(s)
    cdef int N = len(t)
    cdef np.ndarray[np.float_t, ndim=3] K_prime = np.ones((k, M+1, N+1), dtype=np.float)
    cdef np.ndarray[np.float_t, ndim=3] K_dprime = np.zeros((k, M+1, N+1), dtype=np.float)
    cdef int i, m, n

    for i in range(1, k):
        for m in range(M+1):
            for n in range(N+1):
                if min(m, n) < i:
                    K_prime[i, m, n] = 0
                    continue

                if s[m-1] != t[n-1]:
                    K_dprime[i, m, n] = l*K_dprime[i, m, n-1]
                else:
                    K_dprime[i, m, n] = l*(K_dprime[i, m, n-1] + l*K_prime[i-1, m-1, n-1])

                K_prime[i, m, n] = l*K_prime[i, m-1, n] + K_dprime[i, m, n]

    return K_prime

def _find_all_char_indices(ch, string):
    return [idx for idx, ltr in enumerate(string) if ltr == ch]