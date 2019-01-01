Tutorial
========

Contours
--------
cxroots is not a global rootfinder: it can only find roots within a given region of the complex plane.
So the first step is to specify this region as the interior of a contour.
cxroots allows the user to choose from one of four kinds of contours:

.. Would be good if images were hyperlinks to the relevant sections of documentation but this seems impossible.

.. |circlefig| image:: circle.png 

.. |annulusfig| image:: annulus.png 

.. |rectanglefig| image:: rectangle.png 

.. |annulussefig| image:: annulussefig.png 

.. +------------------------+------------------------+
.. | :ref:`circle`          | :ref:`annulus`         |
.. +------------------------+------------------------+
.. | |circlefig|            | |annulusfig|           |
.. +------------------------+------------------------+
.. | :ref:`rectangle`       | :ref:`Annulus Sector`  |
.. +------------------------+------------------------+
.. | |rectanglefig|         | |annulussefig|         |
.. +------------------------+------------------------+

+------------------------+------------------------+------------------------+------------------------+
| :ref:`circle`          | :ref:`rectangle`       | :ref:`annulus`         | :ref:`Annulus Sector`  |
+------------------------+------------------------+------------------------+------------------------+
| |circlefig|            | |rectanglefig|         | |annulusfig|           | |annulussefig|         |
+------------------------+------------------------+------------------------+------------------------+

For example, to define a rectangle whose verticies are the points :math:`0, i, 2+i, 2` we would write:

.. code-block:: python

	from cxroots import Rectangle
	rect = Rectangle([0,2], [0,1])

To check that this is what we want we can plot this contour using matplotlib:

.. code-block:: python

	rect.show()

.. plot::
	:include-source: False

	from cxroots import Rectangle
	rect = Rectangle([0,2], [0,1])
	rect.show()

Rootfinding
-----------
To find the roots of a function :math:`f(z)` within a contour :math:`C` we can use the method :py:meth:`C.roots(f) <cxroots.Contour.Contour.roots>` or preferably :py:meth:`C.roots(f, df) <cxroots.Contour.Contour.roots>` if the derivative `df` is known.

For example, suppose we want to find all the roots of the function :math:`f(z) = iz^5 + z\sin(z)` within a circle of radius 2 and centered at :math:`z=0`.
With cxroots this acomplished with the following short Python script:

.. literalinclude:: docex_tutorial.py
	:lines: 1-7

In the first three lines we define the function :math:`f(z)` and its derivative :math:`f'(z)`. 
We then define our contour, in this case the circle :math:`C=\{z\in\mathbb{C}\,|\,|z|=2\}`.
The method :py:meth:`C.roots(f, df) <cxroots.Contour.Contour.roots>` on the last line returns a :py:class:`~cxroots.RootResult.RootResult` object which we can use to print the roots and their multiplicities:

.. code-block:: python

	print(r)

.. program-output:: python3 -c "from docex_tutorial import *; print(r)"

We can also plot the roots using matplotlib:

.. code-block:: python

	r.show()

.. image:: tutorial_roots.png
	:width: 400px

The :py:class:`RootResult <cxroots.RootResult.RootResult>` object also contains the roots and multiplicites as lists which can be accessed as:

.. code-block:: python

	roots, multiplicities = r
	print(roots)
	print(multiplicities)

.. program-output:: python3 -c "from docex_tutorial import *; roots, multiplicities = r; print(roots); print(multiplicities)"

or as attributes:

.. code-block:: python

	r.roots

.. program-output:: python3 -c "from docex_tutorial import *; print(r.roots)"

.. code-block:: python

	r.multiplicities

.. program-output:: python3 -c "from docex_tutorial import *; print(r.multiplicities)"


.. _Python: http://www.python.org/
