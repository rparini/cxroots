#!/usr/bin/env python
import os
import sys
import shutil
import unittest
from distutils.core import setup, Command
from numpy.distutils.misc_util import get_numpy_include_dirs
from setuptools.command.test import test as TestCommand # Need for 'test' command to be recognised

packages = ['cxroots', 'cxroots.tests']

# get the version
exec(open('cxroots/version.py').read())

# read the README_pip.rst
try:
    with open('README_pip.rst') as file:
        long_description = file.read()
except:
    long_description = None

setup(
    name = 'cxroots',
    version = __version__,
    description = 'Find all the roots (zeros) of a complex analytic function within a given contour in the complex plane.',
    long_description = long_description,
    author = 'Robert Parini',
    author_email = 'robert.parini@gmail.com',
    url = 'https://rparini.github.io/cxroots/',
    license = 'BSD',
    data_files = [("", ["LICENSE"])],
    packages = packages,
    platforms = ['all'],
    dependency_links=['git+git://github.com/pbrod/numdifftools'],
    install_requires = ['pytest-runner', 'numpy', 'scipy', 'docrep', 'mpmath', 'numdifftools'],
    tests_require=['pytest'],
    keywords='roots zeros complex analytic functions',
    classifiers=[
	    'Development Status :: 4 - Beta',
	    'Intended Audience :: Science/Research',
	    'Topic :: Scientific/Engineering :: Mathematics',
	    'License :: OSI Approved :: BSD License',
	    'Programming Language :: Python :: 2.7',
	    'Programming Language :: Python :: 3',
	],
)
