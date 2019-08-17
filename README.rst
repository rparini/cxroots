|pkg_img| |travis| |pyup|

.. |travis| image:: https://travis-ci.org/rparini/cxroots.svg?branch=master 
    :target: https://travis-ci.org/rparini/cxroots/branches
    
.. |pkg_img| image:: https://badge.fury.io/py/cxroots.svg
    :target: https://badge.fury.io/py/cxroots

.. |pyup| image:: https://pyup.io/repos/github/rparini/cxroots/shield.svg
    :target: https://pyup.io/repos/github/rparini/cxroots/

cxroots
=======

cxroots is a Python package for finding all the roots of a function, *f(z)*, of a single complex variable within a given contour, *C*, in the complex plane.  It requires only that:

-  *f(z)* has no roots or poles on *C*
-  *f(z)* is analytic in the interior of *C*

The implementation is primarily based on [KB]_ and evaluates contour integrals involving *f(z)* and its derivative *f'(z)* to determine the roots.  If *f'(z)* is not provided then it is approximated using a finite difference method.  The roots are further refined using Newton-Raphson if *f'(z)* is given or Muller's method if not.  See the `documentation <https://rparini.github.io/cxroots/>`_ for a more details and a tutorial.

With `Python <http://www.python.org/>`_ installed you can install cxroots by entering in the terminal/command line

.. code:: bash

    pip install cxroots

Example
-------

.. code:: python

    from numpy import exp, cos, sin
    f = lambda z: (exp(2*z)*cos(z)-1-sin(z)+z**5)*(z*(z+2))**2
    
    from cxroots import Circle
    C = Circle(0,3)
    roots = C.roots(f)
    roots.show()


.. Relative images do not display on pypi
.. image:: https://github.com/rparini/cxroots/blob/master/README_resources/readmeEx.png?raw=true

.. code:: python

    print(roots)


.. literalinclude readmeExOut.txt doesn't work on github
.. code::

	 Multiplicity |               Root              
	------------------------------------------------
	      2       | -2.000000000000 +0.000000000000i
	      1       | -0.651114070264 -0.390425719088i
	      1       | -0.651114070264 +0.390425719088i
	      3       |  0.000000000000 +0.000000000000i
	      1       |  0.648578080954 -1.356622683988i
	      1       |  0.648578080954 +1.356622683988i
	      1       |  2.237557782467 +0.000000000000i


See also
--------

The Fortran 90 package `ZEAL <http://cpc.cs.qub.ac.uk/summaries/ADKW>`_ is a direct implementation of [KB]_.

Citing cxroots
--------------

  \R. Parini. *cxroots: A Python module to find all the roots of a complex analytic function within a given contour* (2018), https://github.com/rparini/cxroots 

BibTex:

.. code::

	@misc{cxroots,
	  author = {Robert Parini},
	  title = {{cxroots: A Python module to find all the roots of a complex analytic function within a given contour}},
	  url = {https://github.com/rparini/cxroots},
	  year = {2018--}
	}

----------

References
----------

.. [KB] \P. Kravanja and M. Van Barel.  *Computing the Zeros of Analytic Functions*. Springer, Berlin, Heidelberg, 2000.


