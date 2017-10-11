Getting Started
===============

Installation
------------
cxroots requires the Python_ programming language and is compatible with both Python 2 and 3.

With Python installed the easiest way to install cxroots, along with its dependencies, is using the Python Package Index (PyPi) by running on the command line::

	pip install cxroots


Alternatively, cxroots can be downloaded directly from github_.

Rootfinding
-----------
cxroots is not a global rootfinder: it can only find roots within a given region in the complex plane.
The first step for the user is to specify this region as the interior of one of four kinds of contours:


.. Would be good if images were hyperlinks to the relevant sections of documentation but this seems impossible.

.. |circlefig| image:: circle.pdf 

.. |annulusfig| image:: annulus.pdf 

.. |rectanglefig| image:: rectangle.pdf 

.. |annulussefig| image:: annulussefig.pdf 

+------------------------+------------------------+
| :ref:`circle`          | :ref:`annulus`         |
+------------------------+------------------------+
| |circlefig|            | |annulusfig|           |
+------------------------+------------------------+
| :ref:`rectangle`       | :ref:`Annulus Sector`  |
+------------------------+------------------------+
| |rectanglefig|         | |annulussefig|         |
+------------------------+------------------------+

Suppose all the roots of :math:`f(z) = iz^5 + z\sin(z)` within a circle of radius 2 and centered at :math:`z=0` are sought.
The corresponding script would be:

.. literalinclude:: docex_gettingstarted.py

For the full list of arguments for the `.roots(f,df)` method see :ref:`Root Finding`.
The `.roots(f,df)` method returns a :ref:`Root Viewing` object which contains lists of the roots and multiplicites as lists as attributes:

.. literalinclude:: docex_gettingstarted_a.py
	:lines: 3

.. program-output:: python3 docex_gettingstarted_a.py

.. literalinclude:: docex_gettingstarted_b.py
	:lines: 3

.. program-output:: python3 docex_gettingstarted_b.py

For other options for displaying and plotting the found roots see: :ref:`Root Viewing`.


.. _Python: http://www.python.org/
.. _github: https://github.com/rparini/cxroots