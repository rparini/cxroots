from __future__ import division
import numpy as np
from numpy import inf, pi
import scipy.integrate
import math

from cxroots.Contours import Circle, Rectangle

def CxDeriv(f, contour=None):
	"""
	Compute derivaive of an analytic function using Cauchy's Integral Formula for Derivatives
	"""
	if contour is None:
		C = lambda z0: Circle(z0, 1e-3)
	else:
		C = lambda z0: contour

	def df(z0, n):
		integrand = lambda z: f(z)/(z-z0)**(n+1)
		return C(z0).integrate(integrand) * math.factorial(n)/(2j*pi)

	return np.vectorize(df)
