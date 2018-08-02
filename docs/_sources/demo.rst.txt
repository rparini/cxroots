Demonstrating Rootfinding
=========================

The rootfinding process can be visualised using the :meth:`~cxroots.Contour.Contour.demo_roots` method.

.. code-block:: python

	from cxroots import Circle
	from numpy import exp, cos, sin
	f = lambda z: (z*(z+2))**2 * (exp(2*z)*cos(z)-1-sin(z)+z**5)
	C = Circle(0,3)

	C.demo_roots(f)

This will create a matplotlib window and pressing the space bar will move the rootfinding process forward one step by either subdividing a contour or finding the roots within it.

We can also save this process as an animation (in this case a gif) using

.. plot:: 

	from cxroots import Circle
	from numpy import exp, cos, sin
	f = lambda z: (z*(z+2))**2 * (exp(2*z)*cos(z)-1-sin(z)+z**5)
	C = Circle(0,3)

	C.demo_roots(f, saveFile='rootsDemo.gif', writer='imagemagick')

.. image:: rootsDemo.gif
	:width: 400px
