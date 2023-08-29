#!/usr/bin/env python
from setuptools import setup

packages = ["cxroots", "cxroots.tests", "cxroots.contours"]

# get the version, this will assign __version__
with open("cxroots/version.py") as f:
    exec(f.read())  # nosec

# read the README_pip.rst
try:
    with open("README.rst") as file:
        long_description = file.read()
except IOError:
    long_description = None

setup(
    name="cxroots",
    version=__version__,  # noqa
    description="Find all the roots (zeros) of a complex analytic function within a "
    "given contour in the complex plane.",
    long_description=long_description,
    author="Robert Parini",
    author_email="robert.parini@gmail.com",
    url="https://rparini.github.io/cxroots/",
    license="BSD-3-Clause",
    data_files=[("", ["LICENSE"])],
    packages=packages,
    package_data={"cxroots": ["py.typed"]},
    zip_safe=False,  # prevent cxroots from installing as a .egg zip file
    platforms=["all"],
    python_requires=">=3.8",
    setup_requires=["pytest-runner"],
    install_requires=[
        "numpy<1.25",
        "scipy",
        "numpydoc",
        "mpmath",
        "numdifftools>=0.9.39",
        "rich",
    ],
    tests_require=["pytest"],
    extras_require={"plot": ["matplotlib"]},
    keywords="roots zeros complex analytic functions",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
