from __future__ import division
import numpy as np
from numpy import inf, pi
import scipy.integrate
import warnings
from ContArg import contArg

def count_enclosed_roots_arg(C, f, reqEqualZeros=3):
	r"""
	Note: this function is not currently used by the module but is kept for experimentation.

	Return the number of roots of f(z) (counting multiplicities) which lie within C.  
	The number of zeros is computed as the difference in the argument of f(z) continued
	around the contour C [DL,BVP].  

	This has the potential to be unreliable since this method will always find that the 
	number of zeros is an integer, rather than smoothly converging to an integer as the 
	number of points at which f is sampled increases.  This makes it difficult to tell 
	when the number of zeros has genuinely converged.  For this reason it seems preferable
	to approximate f'(z) if not provided by the user.

	In this case the number of sample points is doubled until the last reqEqualZeros 
	(3 by default) evaluations of the number of zeros are equal and non-negative. 
	The contour is also rejected as being unreliable if at any point :math:`|df/f| > 10^6`.

	Parameters
	----------
	C : Contour
		The enclosed_roots_arg function returns the number of roots of f(z) within C
	f : function
		Function of a single variable f(x)
	reqEqualZeros : int, optional
		If the Cauchy integral is computed by continuing the argument around the contour (ie. if df is None)
		then the routine requires that the last reqEqualZeros evaluations of the number of enclosed zeros 
		are equal and non-negative.  Default is 3.

	Returns
	-------
	int
		The number of zeros of f (counting multiplicities) which lie within the contour
	
	References
	----------
	[DSZ] "Locating all the Zeros of an Analytic Function in one Complex Variable" 
		M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2

	[BVP] Gakhov, F. D. "Boundary value problems", section 12 (2014), Elsevier.
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		N = 25 # initial number of sample points

		t = np.linspace(0,1,N+1)
		z = self(t)
		vals = [f(z)]
		arg = contArg(vals[-1])
		I = [int((arg[-1] - arg[0])/(2*pi))]

		while len(I) < reqEqualZeros or not np.all(np.array(I[-reqEqualZeros:-1])==I[-1]) or I[-1] < 0:
			N = 2*N
			t = np.linspace(0,1,N+1)
			z = self(t)

			val = np.zeros_like(t, dtype=np.complex128)
			val[::2] = vals[-1]
			val[1::2] = f(z[1::2]) # only compute f at new points
			vals.append(val)

			arg = contArg(val)
			I.append(int((arg[-1] - arg[0])/(2*pi)))

		numberOfZeros = I[-1]

		# reject the result if the contour is believed to be on or very close to a root
		# From [DL] |f'(z)/f(z)| is of the order of (distance to nearest zero)^{-1}
		df = np.diff(val)/np.diff(z)
		if np.any(abs(df/val[:-1]) > 10):
			# the contour might be too close to a root.  To determine this
			# get a greater resolution in the area where abs(df/val[:-1]) is large

			# try and converge to the true value of |df/f| to determine if it's too big
			maxdff = []
			while len(maxdff) < 2 or abs(maxdff[-2] - maxdff[-1]) > 10:
				dff = abs(df/val[:-1])
				maxdff.append(max(dff))

				if maxdff[-1] > 1e6:
					raise RuntimeError('Contour is believed to be on a very close to a root')

				else:
					# zoom in further to see if |df/f| grows
					maxArg = np.argmax(dff)
					t = np.linspace(t[maxArg-1], t[maxArg+1], 101)
					z = self(t)
					val = f(z)
					df = np.diff(val)/np.diff(z)

	return numberOfZeros

