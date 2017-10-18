#!/usr/bin/env python
import os
import sys
import shutil
import unittest
from distutils.core import setup, Command
from numpy.distutils.misc_util import get_numpy_include_dirs

packages = ['cxroots', 'cxroots.tests']

# get the version
exec(open('cxroots/version.py').read())

# read the README_pip.rst
try:
    with open('README_pip.rst') as file:
        long_description = file.read()
except:
    long_description = None

# create test commmand
class TestCommand(Command):
    # See: https://justin.abrah.ms/python/setuppy_distutils_testing.html by Justin Abrahms
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys, subprocess

        raise SystemExit(
            subprocess.call([sys.executable,
                             '-m',
                             'unittest',
                             'discover',
                             '-v',
                             'cxroots/tests']))

setup(
    name = 'cxroots',
    version = __version__,
    description = 'Find all the roots (zeros) of a complex analytic function within a given contour in the complex plane.',
    long_description = long_description,
    author = 'Robert Parini',
    author_email = 'rp910@york.ac.uk',
    url = 'https://rparini.github.io/cxroots/',
    license = 'BSD',
    packages = packages,
    platforms = ['all'],
    install_requires = ['numpy', 'scipy', 'docrep', 'mpmath'],
    keywords='roots zeros complex analytic functions',
    classifiers=[
	    'Development Status :: 4 - Beta',
	    'Intended Audience :: Science/Research',
	    'Topic :: Scientific/Engineering :: Mathematics',
	    'License :: OSI Approved :: BSD License',
	    'Programming Language :: Python :: 2.7',
	    'Programming Language :: Python :: 3',
	],
    cmdclass={
        'test': TestCommand,
    }
)
