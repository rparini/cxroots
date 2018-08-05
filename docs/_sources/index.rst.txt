cxroots: rootfinding for complex analytic functions
===================================================

cxroots is a Python library for finding all the roots of a complex analytic function of a single complex variable within a given contour in the complex plane. 
The implementation is based on [1].

 .. and the mathematics of this is discussed in greater detail in :ref:`Theory`

The cxroots code is hosted at `GitHub <https://github.com/rparini/cxroots>`_ and is open source under a `BSD 3-Clause License <https://github.com/rparini/cxroots/blob/master/LICENSE>`_


Documentation
-------------
.. toctree::
	:caption: Getting Started

	installation.rst
	tutorial.rst

.. toctree::
	:caption: User Guides

	demo.rst
	guesses.rst

.. toctree::
	:caption: Examples

	ex_annular_combustion.rst

.. toctree::
	:caption: Reference

	paths.rst
	contours.rst
	prod.rst
	iteration.rst
	result.rst


References
----------
[1] Peter Kravanja and Marc Van Barel. "Computing the Zeros of Analytic Functions". Springer Berlin Heidelberg, 2000.
