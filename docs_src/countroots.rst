Counting Roots
==============

Sometimes we don't need to know the precise location of every root, just how many there are.
We can use the :meth:`~cxroots.contour.Contour.count_roots` method to just count the number of roots of an analytic function within a given contour.

The example below counts all the roots of :math:`f(z)=z^2(z+2)^2(z+4)^2` within the circle :math:`|z|<3`.

.. literalinclude:: countroots.py
	:language: python

.. program-output:: python3 countroots.py
