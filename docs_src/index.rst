cxroots: rootfinding for complex analytic functions
===================================================

cxroots is a Python package for finding all the roots of a function, :math:`f(z)`, of a single complex variable within a given contour, :math:`C`, in the complex plane.  It requires only that both:

-  :math:`f(z)` has no roots or poles on :math:`C`
-  :math:`f(z)` is analytic in the interior of :math:`C`

The implementation is primarily based on :cite:t:`Kravanja2000a` and evaluates contour integrals involving :math:`f(z)` and its derivative :math:`f'(z)` to approximate the roots. Then iterative methods, such as Newton-Raphson or Muller's method are used to refine the roots. If :math:`f'(z)` is not provided then it is approximated. See the :ref:`theory:theory` page for a more detailed explanation.

.. code:: python

    from numpy import exp, cos, sin
    f = lambda z: (exp(2*z)*cos(z)-1-sin(z)+z**5)*(z*(z+2))**2
    
    from cxroots import Circle
    C = Circle(0,3)
    roots = C.roots(f)
    roots.show()

.. image:: https://github.com/rparini/cxroots/blob/master/README_resources/readme_example.png?raw=true


Documentation
-------------
.. toctree::
	:caption: Getting Started

	installation.rst
	tutorial.rst
	changelog.rst
	theory.rst

.. toctree::
	:caption: User Guides

	countroots.rst
	demo.rst
	guesses.rst
	logging.rst

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
