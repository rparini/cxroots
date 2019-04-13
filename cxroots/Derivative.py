from __future__ import division
import math

import numpy as np
from numpy import pi

import numdifftools.fornberg as ndf

@np.vectorize
def CxDerivative(f, z0, n=1, contour=None, absIntegrationTol=1e-10, verbose=False):
	r"""
	Compute the derivaive of an analytic function using Cauchy's
	Integral Formula for Derivatives.

	.. math::

		f^{(n)}(z_0) = \frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z-z_0)^{n+1}} dz

	Parameters
	----------
	f : function
		Function of a single variable f(x).
	z0 : complex
		Point to evaluate the derivative at.
	n : int
		The order of the derivative to evaluate.
	contour : :class:`Contour <cxroots.Contour.Contour>`, optional
		The contour, C, in the complex plane which encloses the point z0.
		By default the contour is the circle |z-z_0|=1e-3.
	absIntegrationTol : float, optional
		The absolute tolerance required of the integration routine.
	verbose : bool, optional
		If True runtime information will be printed.  False be default.

	Returns
	-------
	f^{(n)}(z0) : complex
		The nth derivative of f evaluated at z0
	"""
	if contour is None:
		from .contours.Circle import Circle
		C = lambda z0: Circle(z0, 1e-3)
	else:
		C = lambda z0: contour

	integrand = lambda z: f(z)/(z-z0)**(n+1)
	integral = C(z0).integrate(integrand, absTol=absIntegrationTol, verbose=verbose)
	return integral * math.factorial(n)/(2j*pi)


def find_multiplicity(root, f, df=None, rootErrTol=1e-10, verbose=False):
	"""
	Find the multiplicity of a given root of f by computing the
	derivatives of f, f^{(1)}, f^{(2)}, ... until
	|f^{(n)}(root)|>rootErrTol.  The multiplicity of the root is then
	equal to n.  The derivative is calculated with `numdifftools <http://numdifftools.readthedocs.io/en/latest/api/numdifftools.html#numdifftools.fornberg.derivative>`_
	which employs a method due to Fornberg.

	Parameters
	----------
	root : complex
		A root of f, f(root)=0.
	f : function
		An analytic function of a single complex variable such that
		f(root)=0.
	df : function, optional
		The first derivative of f.  If not known then df=None.
	contour : Contour, optional
		The integration contour used to evaluate the derivatives.
	rootErrTol : float, optional
		It will be assumed that f(z)=0 if numerically |f(z)|<rootErrTol.
	verbose : bool, optional
		If True runtime information will be printed.  False be default.

	Returns
	-------
	multiplicity : int
		The multiplicity of the given root.
	"""
	if abs(f(root)) > rootErrTol:
		raise ValueError("""
			The provided 'root' is not a root of the given function f.
			Specifically, %f = abs(f(root)) > rootErrTol = %f
			"""%(abs(f(root)), rootErrTol))

	n = 1
	while True:
		if df is not None:
			if n==1:
				err = abs(df(root))
			else:
				# ndf.derivative returns an array [f, f', f'', ...]
				err = abs(ndf.derivative(df, root, n-1)[n-1])
		else:
			err = abs(ndf.derivative(f, root, n)[n])

		if verbose:
			print('n', n, '|df^(n)|', err)

		if err > rootErrTol:
			break

		n += 1

	return n

