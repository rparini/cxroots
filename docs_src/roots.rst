Root Finding
============

Overview
--------

The roots within a :ref:`Contour <Contours>` are found by calling the method:

.. automethod:: cxroots.Contours.Contour.roots

The *Contour.roots* method returns a :ref:`rootResult <Root Viewing>` object which stores the roots and their multiplicites and provides convienent methods for displaying them.

.. automethod:: cxroots.Contours.Contour.demo_roots


Guessing Roots
--------------

The *guessRoots* argument of :func:`Contour.roots <cxroots.Contours.Contour.roots>` can be used to supply any roots or approximations to roots the user already knows.
The *guessRoots* argument should be a list of roots or, if the multiplicity is known, a list of (root, multiplicity) tuples.

.. literalinclude:: docex_guessRoots.py
	:lines: -7

.. program-output:: python3 docex_guessRoots.py

The roots are immediately checked and recorded by cxroots

.. literalinclude:: docex_guessRoots.py
	:lines: 9

.. image:: guessRoots.gif
	:width: 400px


Guessing Root Symmetry
----------------------

It may be that something is known about the structure of the roots.
The rootfinder can be told this using the *guessRootSymmetry* argument which should be a function of a complex number, :math:`z`, which returns a list of roots assuming that :math:`z` is a root. 

For example, if :math:`z_i` is a root of 

.. math::
	
	f(z)=z^{26}-2z^{10}+\frac{1}{2}z^6-1

then so is :math:`\overline{z_i}` and :math:`-z`.

.. literalinclude:: docex_guessRootSymmetry.py
	:lines: -7

.. image:: ex_rootSymmetry.gif
	:width: 400px

Using guessRootSymmetry can save some time:

.. literalinclude:: docex_guessRootSymmetry.py
	:lines: 10-18

.. program-output:: python3 docex_guessRootSymmetry.py

