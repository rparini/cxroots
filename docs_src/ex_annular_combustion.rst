Stability of an Annular Combustion Chamber
===================================================

In section 4.2 of [DSZ]_ the problem of stability of an annular combustion chamber was used as a test case for their rootfinding algorthim.  The problem consisted of finding all the zeros of the holomorphic function

.. math::

	f(z)=z^2+Az+Be^{-Tz}+C

with constants

.. math::
	A=-0.19435\,,\quad
	B=1000.41\,,\quad
	C=522463\,,\quad
	T=0.005\,.

The result there can be replicated in cxroots with:

.. plot::
	:include-source: True

	from numpy import exp
	from cxroots import Rectangle

	A = -0.19435
	B = 1000.41
	C = 522463
	T = 0.005

	f = lambda z: z**2 + A*z + B*exp(-T*z) + C
	df = lambda z: 2*z + A - B*T*exp(-T*z)

	rectangle = Rectangle([-15000,5000], [-15000,15000])
	roots = rectangle.roots(f, df)
	roots.show()


References
----------
.. [DSZ] "Locating all the Zeros of an Analytic Function in one Complex Variable" M. Dellnitz, O. Schütze, Q. Zheng, J. Compu. and App. Math. (2002), Vol. 138, Issue 2
