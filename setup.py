#!/usr/bin/env python
import os
import sys
import shutil
import unittest
from distutils.core import setup, Command
from distutils.extension import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

packages = ['cxroots', 'cxroots.tests']

# get the version
exec(open('cxroots/version.py').read())

setup(
    name = 'cxroots',
    version = __version__,
    description = 'Find all the roots of a function within a contour in the complex plane',
    author = 'Robert Parini',
    url = 'https://github.com/rparini/cxroots',
    license = 'BSD',
    packages = packages,
    platforms = ['all'],
    install_requires = ['numpy', 'scipy', 'docrep'],
    classifiers=[
	    'Development Status :: 4 - Beta',
	    'Intended Audience :: Science/Research',
	    'Topic :: Scientific/Engineering :: Mathematics',
	    'License :: OSI Approved :: BSD License',
	    'Programming Language :: Python :: 2.7',
	    'Programming Language :: Python :: 3',
	],
	keywords='roots zeros complex analytic functions',
	
)
