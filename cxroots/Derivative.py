from __future__ import division
import numpy as np
from numpy import inf, pi
import math

def CxDerivative(f, n=1, contour=None):
	"""
	Compute the derivaive of an analytic function using Cauchy's Integral Formula for Derivatives
	"""
	if contour is None:
		from .Contours.Circle import Circle
		C = lambda z0: Circle(z0, 1e-3)
	else:
		C = lambda z0: contour

	def df(z0):
		integrand = lambda z: f(z)/(z-z0)**(n+1)
		return C(z0).integrate(integrand) * math.factorial(n)/(2j*pi)

	return np.vectorize(df)

def get_multiplicity(f, root, contour=None, df=None, rootErrTol=1e-12):
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
			if n==1 and abs(df(root)) > rootErrTol:
				break
			elif abs(CxDerivative(df,n-1,contour)(root)) > rootErrTol:
				break

		if abs(CxDerivative(f,n,contour)(root)) > rootErrTol:
			break

	return n

