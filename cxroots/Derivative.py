import numpy as np
from numpy import inf, pi
import math

@np.vectorize
def CxDerivative(f, z0, n=1, contour=None, absIntegrationTol=1e-10, verbose=False):
	"""
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
	absIntegrationTol : float, optional
		The absolute tolerance required of the integration routine.
	contour : Contour, optional
		The contour, C, in the complex plane which encloses the point z0.
		By default the contour is the circle |z-z_0|=1e-3.
	verbose : Bool, optional
		If True information about the progress of the contour 
		integration will be printed.  False by default.

	Returns
	-------
	f^{(n)}(z0) : complex
		The nth derivative of f evaluated at z0
	"""
	if contour is None:
		from .Contours.Circle import Circle
		C = lambda z0: Circle(z0, 1e-3)
	else:
		C = lambda z0: contour

	integrand = lambda z: f(z)/(z-z0)**(n+1)
	integral = C(z0).integrate(integrand, absTol=absIntegrationTol, verbose=verbose)
	return integral * math.factorial(n)/(2j*pi)

def get_multiplicity(f, root, contour=None, df=None, rootErrTol=1e-10, verbose=False):
	"""
	Find the multiplicity of a given root of f.
	
	Parameters
	----------
	f : function
		An analytic function of a single complex variable such that
		f(root)=0.
	root : complex
		A root of f, f(root)=0.
	df : function, optional
		The first derivative of f.  If not known then df=None
	contour : 

	rootErrTol : float
		It will be assumed that f(z)=0 if numerically |f(z)|<rootErrTol.

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
				err = abs(CxDerivative(df, root, n-1, contour, rootErrTol, verbose))
		else:
			err = abs(CxDerivative(df, root, n, contour, rootErrTol, verbose))

		if verbose:
			print('n', n, '|df^(n)|', err)

		if err > rootErrTol:
			break

		n += 1

	return n

