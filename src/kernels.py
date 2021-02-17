# -*- coding: utf-8 -*-
"""
Implementation of the useful kernels for our classifier


Numerical kernels
    - linear kernel
    - polynomial kernel
    - gaussian kernel

String kernels
    - kernel spectrum
    - weighted degree kernel
    - substring kernel (implemented using cython)
"""

import numpy as np
import ssk_c


######################### Linear kernels #########################

def linear_kernel(x, y):
    """
    Linear kernel K(x,y) = <x,y>
    """
    return np.dot(x, y)


def polynomial_kernel(x, y, power=3):
    """
    Polynomial kernel K(x,y) = (1+<x,y>)^p
    """
    return (1 + np.dot(x, y))**power


def gaussian_kernel(x, y, sigma=1):
    """
    Gaussian kernel K(x,y) = exp(-|x-y|^2/(2sigma^2))
    """
    return np.exp(-np.linalg.norm(x-y, ord=2)**2/(2*sigma**2))



######################### Spectrum kernel #########################

def get_substring(string, length):
    """
    Look for the substring of size length of the string
    """
    return set(string[i:i+length] for i in range(len(string)-length+1))


def kernel_spectrum(string1, string2, spectrum=3):
    """
    Compute the kernel spectrum between two strings

    Parameters:
        - string1: string
        - string2: string
        - spectrum: int, default 3, length of the substring
    """
    substring1 = get_substring(string1, spectrum)
    substring2 = get_substring(string2, spectrum)

    substrings = list(set(substring1) | set(substring2))

    kernel = 0

    for substring in substrings:
        kernel += string1.count(substring)*string2.count(substring)

    return kernel


######################### Weight degree kernel #########################

def weight_degree(string1, string2, degree):
    """
    Compute the weight degree kernel between two strings

    Parameters:
        - string1: string
        - string2: string
        - degree: int, degree of the kernel
    """
    length = len(string1)
    res = 0
    for k in range(1, degree + 1):
        beta_k = 2 * (degree - k + 1) / degree / (degree + 1)
        c_st = 0
        for l in range(1, length - k + 1):
            c_st += (string1[l:l + k] == string2[l:l + k])
        res += beta_k * c_st
    return res


######################### SSK #########################

def ssk_kernel(string1, string2, params):
    """
    Compute the substring kernel between two strings
    Uses a cython function due to the computational time

    Parameters:
        - string1: string
        - string2: string
        - params: parameters of the kernel
                  (length of the substring and weight parameter)
    """
    k, l = params
    return ssk_c.ssk_cython(string1, string2, k, l)



######################### Kernel selection #########################

def select_method(method):
    """
    Select the kernel and its parameter from the string method
    """
    method = method.split('_')

    if method[0] == 'KS':
        kernel = kernel_spectrum
        params = int(method[1])

    elif method[0] == 'WD':
        kernel = weight_degree
        params = int(method[1])

    elif method[0] == 'SSK':
        kernel = ssk_kernel
        params = [int(method[1]), float(method[2])]

    else:
        raise ValueError('Method not implemented')
    return kernel, params
