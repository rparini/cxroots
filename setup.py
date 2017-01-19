#!/usr/bin/env python
import os
import sys
import shutil
import unittest
from distutils.core import setup, Command
from distutils.extension import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

packages = ['cxroots']

# configure setup
setup(
    name = 'cxroots',
    description = 'Find all the roots of a function within a contour in the complex plane',
    author = 'Robert Parini',
    packages = packages,
    platforms = ['all']
)
