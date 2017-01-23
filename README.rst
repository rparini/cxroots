
cxroots
=======

A Python module to compute all the (simple) zeros of an analytic
function within a given contour.

--------------

Introduction
------------

Given a contour :math:`C` in the complex plane and a function
:math:`f(z)` (and optionally it's derivative :math:`f'(z)`) which:

-  Has no roots or poles on :math:`C`
-  Is analytic in the interior of :math:`C`
-  Has only simple roots in the interior of :math:`C`

the module is able to compute all the roots of within :math:`C`.

.. code:: python

    from cxroots import Rectangle, showRoots, findRoots, demo_findRoots
    from numpy import sin, cos
    
    rect = Rectangle([-1.5,1.5],[-2,2])
    f  = lambda z: z**10 - 2*z**5 + sin(z)*cos(z/2)
    showRoots(rect, f)



.. image:: README_files/README_1_0.png


The implementation is primarily based on [1] where the number of roots
within a contour, :math:`N`, is calculated by numerical integration of
the Cauchy integral,

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?N=\frac{1}{2i\pi}\oint_C\frac{f'(z)}{f(z)}dz."/>
</p>

The original contour is subdivided until each sub-contour only contains
a single root and then the Newton-Raphson method is repeatedly used with
random startpoints until the root within each sub-contour is found.

If :math:`f'(z)` is not provided by the user then it is approximated
with a Taylor expansion as in [2].

Future improvements:
~~~~~~~~~~~~~~~~~~~~

-  Allow for multiplicities of roots within :math:`C`. Perhaps using the
   method of formal orthogonal polynomials suggested in [3]
-  Approximate the roots of :math:`f` within a contour as the zeros of a
   constructed polynomial [2]
-  If :math:`f'(z)` is not provided then use the approximation to
   :math:`f'(z)` in the Newton-Raphson method

Installation
~~~~~~~~~~~~

Can be downloaded from GitHub and installed by running the included
'setup.py' with

.. code:: bash

    python setup.py install

Documentation
~~~~~~~~~~~~~

For a tutorial on the use of this module and a discrption of how it
works see the `documentation <docs/main.ipynb>`__.

--------------

References
^^^^^^^^^^

[1] M. Dellnitz, O. Schütze and Q. Zheng, "Locating all the Zeros of an
Analytic Function in one Complex Variable" J. Compu. and App. Math.
(2002) Vol. 138, Issue 2

[2] L.M. Delves and J.N. Lyness, "A Numerical Method for Locating the
Zeros of an Analytic function" Mathematics of Computation (1967) Vol.21,
Issue 100

[3] P. Kravanja, T. Sakurai and M. Van Barel, "On locating clusters of
zeros of analytic functions" BIT (1999) Vol. 39, No. 4
