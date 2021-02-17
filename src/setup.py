# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 19:39:59 2021

@author: clara
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
      ext_modules=cythonize("ssk_c.pyx"), 
      include_dirs=[numpy.get_include()]
      )

# command: python setup.py build_ext --inplace