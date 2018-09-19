cxroots: rootfinding for complex analytic functions
===================================================

cxroots is a Python package for finding all the roots of a function, *f(z)*, of a single complex variable within a given contour, *C*, in the complex plane.  It requires only that:

-  *f(z)* has no roots or poles on *C*
-  *f(z)* is analytic in the interior of *C*

The implementation is primarily based on [KB]_ and evaluates contour integrals involving *f(z)* and its derivative *f'(z)* to determine the roots.  If *f'(z)* is not provided then it is approximated using a finite difference method.  The roots are further refined using Newton-Raphson if *f'(z)* is given or Muller's method if not.

.. code:: python

    from numpy import exp, cos, sin
    f = lambda z: (exp(2*z)*cos(z)-1-sin(z)+z**5)*(z*(z+2))**2
    
    from cxroots import Circle
    C = Circle(0,3)
    roots = C.roots(f)
    roots.show()

.. image:: https://github.com/rparini/cxroots/blob/master/README_resources/readmeEx.png?raw=true


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

.. [KB] \P. Kravanja and M. Van Barel.  *Computing the Zeros of Analytic Functions*. Springer, Berlin, Heidelberg, 2000.


