from __future__ import division
import numpy as np
from numpy import inf, pi
import scipy.integrate
import math

def CxDeriv(f, contour=None):
	"""
	Compute the derivaive of an analytic function using Cauchy's Integral Formula for Derivatives
	"""
	if contour is None:
		from cxroots.Contours import Circle
		C = lambda z0: Circle(z0, 1e-3)
	else:
		C = lambda z0: contour

	def df(z0, n=1):
		integrand = lambda z: f(z)/(z-z0)**(n+1)
		return C(z0).integrate(integrand) * math.factorial(n)/(2j*pi)

	return np.vectorize(df)

def multiplicity_correct(f, df, root, multiplicity, rootErrTol=1e-10):
	"""
	Check a given multplicity by calculating the derivatives
	"""
	if df is None:
		derivs = [f(root)] + [CxDeriv(f)(root,n) for n in range(1,multiplicity+1)]
	else:
		derivs = [f(root), df(root)] + [CxDeriv(f)(root,n) for n in range(2,multiplicity+1)]

	return np.all(np.abs(derivs[:-1]) < rootErrTol) and np.abs(derivs[-1]) > rootErrTol
