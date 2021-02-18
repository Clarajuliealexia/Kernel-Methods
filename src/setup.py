# -*- coding: utf-8 -*-
"""
Usefull file to create librairy for subtring kernel
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
      ext_modules=cythonize("ssk_c.pyx"), 
      include_dirs=[numpy.get_include()]
      )

# command: python setup.py build_ext --inplace
