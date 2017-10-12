.. image:: https://travis-ci.org/rparini/cxroots.svg?branch=master

cxroots
=======

cxroots is a Python package for finding all the roots of a function, *f(z)*, of a single complex variable within a given contour, *C*, in the complex plane.  It requires only that:

-  *f(z)* has no roots or poles on *C*
-  *f(z)* is analytic in the interior of *C*

.. code:: python

    from numpy import exp, cos, sin
    f = lambda z: (z*(z+2))**2 * (exp(2*z)*cos(z)-1-sin(z)+z**5)
    
    from cxroots import Circle
    C = Circle(0,3)
    roots = C.roots(f)
    roots.show()


.. image:: https://github.com/rparini/cxroots/blob/master/README_files/README_1_0.png?raw=true


.. code:: python

    print(roots)


::

     Multiplicity |               Root              
    ------------------------------------------------
          2       | -2.000000000000 -0.000000000000i
          1       | -0.651114070264 -0.390425719088i
          1       | -0.651114070264 +0.390425719088i
          3       |  0.000000000000 +0.000000000000i
          1       |  0.648578080954 -1.356622683988i
          1       |  0.648578080954 +1.356622683988i
          1       |  2.237557782467 +0.000000000000i

See the `documentation <https://rparini.github.io/cxroots/>`_ or `GitHub <https://github.com/rparini/cxroots>`_ page for more details and examples.

Installation
------------

Install on the command line with 

.. code:: bash

    pip install cxroots
