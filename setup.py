#!/usr/bin/env python
from setuptools import setup

packages = ['cxroots', 'cxroots.tests', 'cxroots.contours']

# get the version, this will assign __version__
exec(open('cxroots/version.py').read())

# read the README_pip.rst
try:
    with open('README.rst') as file:
        long_description = file.read()
except IOError:
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
    zip_safe = False,   # prevent cxroots from installing as a .egg zip file
    platforms = ['all'],
    setup_requires = ['pytest-runner'],
    install_requires = ['numpy', 'scipy', 'docrep', 'mpmath', 'numdifftools>=0.9.39'],
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
