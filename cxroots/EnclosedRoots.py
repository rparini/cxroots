from __future__ import division
import numpy as np
from numpy import inf, pi
import scipy.integrate
import warnings

from cxroots.Contours import Circle, Rectangle

def count_enclosed_roots_arg(C, f, reqEqualZeros=3):
	r"""
	Note: this fuction is not currently used by the module but is kept for experimentation.

	Return the number of roots of f(z) (counting multiplicies) which lie within C.  
	The number of zeros is computed as the difference in the argument of f(z) continued
	around the contour C [DL,BVP].  

	This has the potential to be unreliable since this method will always find that the 
	number of zeros is an integer, rather than smoothly converging to an integer as the 
	number of points at which f is sampled increases.  This makes it difficult to tell 
	when the number of zeros has genuinely converged.  For this reason it seems preferable
	to approximate f'(z) using a Taylor expansion if f'(z) it is not provided by the user.

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

def count_enclosed_roots(C, f, df=None, integerTol=0.2, taylorOrder=20):
	r"""
	For a function of one complex variable, f(z), which is analytic in and within the contour C,
	return the number of zeros (counting multiplicities) within the contour calculated, using 
	Cauchy's argument principle, as
	
	.. math::

		\frac{1}{2i\pi} \oint_C \frac{f'(z)}{f(z)} dz.

	If df(z), the derivative of f(z), is provided then the above integral is computed directly.
	Otherwise the derivative is approximated using a Taylor expansion about the central point
	within the contour C.  The Taylor coefficients are calculated in such a way as to reuse
	the function evaluations of f(z) on the contour C, as in method C of [DSZ].

	The number of points on each segment of the contour C at which f(z) and df(z) are sampled 
	starts at 2+1 and at the k-th iteration the number of points is 2**k+1.  At each iteration 
	the above integral is calculated using `SciPy's implementation of the Romberg method <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html>`_.
	The routine exits if the difference between sucessive iterations is < integerTol/2.

	The number of roots is then the closest integer to the final value of the integral
	and the result is only accepted if the final value of the integral is within integerTol
	of the closest integer.  If this is not the case then a RuntimeError is raised.
	
	Parameters
	----------
	C : Contour
		The enclosed_roots function returns the number of roots of f(z) within C
	f : function
		Function of a single variable f(x)
	df : function, optional
		Function of a single variable, df(x), providing the derivative of the function f(x) 
		at the point x.
	integerTol : float, optional
		How close the result of the Romberg integration has to be to an integer for it to be
		accepted (only used if df is given).  The absolute tolerance of the Romberg integration
		will be integerTol/2.
	taylorOrder : int, optional
		The number of terms for the Taylor expansion approximating df, provided df is not 
		already given by user.

	Returns
	-------
	int
		The number of zeros of f (counting multiplicities) which lie within the contour
	
	References
	----------
	[DSZ] "Locating all the Zeros of an Analytic Function in one Complex Variable" 
		M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2
	"""

	N = 1
	fVal  = [None]*len(C.segments)
	dfVal = [None]*len(C.segments)
	I = []

	approx_df = False
	if df is None:
		approx_df = True

	# XXX: define err as the difference between sucessive iterations of the Romberg
	# 	   method for the same number of points?
	while len(I) < 2 or abs(I[-2] - I[-1]) > integerTol/2.:
		N = 2*N
		t = np.linspace(0,1,N+1)
		dt = t[1]-t[0]

		# store new function evaluations
		for i, segment in enumerate(C.segments):
			z = segment(t)
			if fVal[i] is None:
				fVal[i] = f(z)
			else:
				newfVal = np.zeros_like(t, dtype=np.complex128)
				newfVal[::2] = fVal[i]
				newfVal[1::2] = f(z[1::2])
				fVal[i] = newfVal

		if approx_df:
			# use available function evaluations to approximate df
			z0 = C.centerPoint
			a = []
			for s in range(taylorOrder):
				a_s = 0
				for i, segment in enumerate(C.segments):
					integrand = fVal[i]/(segment(t)-z0)**(s+1)*segment.dzdt(t)
					
					# romberg integration on a set of sample points
					a_s += scipy.integrate.romb(integrand, dx=dt)/(2j*pi)
				a.append(a_s)

			df = lambda z: sum([j*a[j]*(z-z0)**(j-1) for j in range(taylorOrder)])

		for i, segment in enumerate(C.segments):
			z = segment(t)
			if dfVal[i] is None:
				dfVal[i] = df(z)
			else:
				newdfVal = np.zeros_like(t, dtype=np.complex128)
				newdfVal[::2] = dfVal[i]
				newdfVal[1::2] = df(z[1::2])
				dfVal[i] = newdfVal

		segment_integrand = [dfVal[i]/fVal[i]*segment.dzdt(t) for i, segment in enumerate(C.segments)]
		segment_integral  = [scipy.integrate.romb(integrand, dx=dt)/(2j*pi) for integrand in segment_integrand]
		I.append(sum(segment_integral))

	numberOfZeros = int(round(I[-1].real))
	if numberOfZeros < 0 or abs(I[-1].real - numberOfZeros) > integerTol or abs(I[-1].imag) > integerTol:
		raise RuntimeError('The integral %s is not sufficiently close to a positive integer'%integral)

	return numberOfZeros


if __name__ == '__main__':
	from numpy import sin, cos
	f  = lambda z: z**10 - 2*z**5 + sin(z)*cos(z/2)
	df = lambda z: 10*(z**9 - z**4) + cos(z)*cos(z/2) - 0.5*sin(z)*sin(z/2)

	rect = Rectangle([-1.5,1.5],[-2,2])
	circle = Circle(0,2)

	# print(rect.enclosed_zeros(f, df))
	# print(rect.enclosed_zeros(f))
	# print(circle.enclosed_zeros(f))

	# print(enclosed_roots(rect, f, df))
	print(enclosed_roots(rect, f))