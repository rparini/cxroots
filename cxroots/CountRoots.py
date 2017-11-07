from __future__ import division
import numpy as np
from numpy import inf, pi
import scipy.integrate
import scipy.misc
import warnings

from .CxDerivative import CxDeriv

def prod(C, f, df=None, phi=lambda z:1, psi=lambda z:1, absTol=1e-12, relTol=1e-12, divMax=10, method='quad', verbose=False):
	r"""
	Compute the symmetric bilinear form used in (1.12) of [KB]

	.. math::

		<\phi,\psi> = \frac{1}{2i\pi} \oint_C \phi(z) \psi(z) \frac{f'(z)}{f(z)} dz.
	
	References
	----------
	[KB] "Computing the zeros of analytic functions" by Peter Kravanja, Marc Van Barel, Springer 2000
	"""
	N = 1
	k = 0
	I = []
	integrandMax = []

	approx_df = False
	if df is None:
		approx_df = True

	# print('prod:', C)

	if method == 'romb':
		import numdifftools.fornberg as ndf
		# XXX: define err as the difference between successive iterations of the Romberg
		# 	   method for the same number of points?
		while (len(I) < 2 or (abs(I[-2] - I[-1]) > absTol and abs(I[-2] - I[-1]) > relTol*abs(I[-1]))) and k < divMax:
			N = 2*N
			t = np.linspace(0,1,N+1)
			k = int(np.log2(len(t)-1))
			dt = t[1]-t[0]

			# compute/retrieve function evaluations
			fVal = np.array([segment.trapValues(f,k) for segment in C.segments])
			phiVal = np.array([phi(segment(t)) for segment in C.segments])
			psiVal = np.array([psi(segment(t)) for segment in C.segments])

			if approx_df:
				### approximate df/dz with finite difference, see: numdifftools.fornberg
				# interior stencil size = 2*m + 1
				# boundary stencil size = 2*m + 2
				m = 1
				dfdt = [ndf.fd_derivative(fx, t, n=1, m=m) for fx in fVal]
				dfVal = [dfdt[i]/segment.dzdt(t) for i, segment in enumerate(C.segments)]

			else:
				dfVal = np.array([segment.trapValues(df,k) for segment in C.segments])

			segment_integrand = [phiVal[i]*psiVal[i]*dfVal[i]/fVal[i]*segment.dzdt(t) for i, segment in enumerate(C.segments)]
			segment_integral = scipy.integrate.romb(segment_integrand, dx=dt, axis=-1)/(2j*pi)
			I.append(sum(segment_integral))

			if verbose:
				if k > 1:
					print(k, 'I', I[-1], 'err', I[-2] - I[-1])
				else:
					print(k, 'I', I[-1])

		return I[-1], abs(I[-2] - I[-1])

	elif method == 'quad':
		if approx_df:
			# XXX: need to find a better way around this
			dx = 1e-8
			df = lambda z: scipy.misc.derivative(f, z, dx=dx, n=1, order=3)
			
			# df = CxDeriv(f) # too slow

		I, err = 0, 0
		for segment in C.segments:
			integrand_cache = {}
			def integrand(t):
				if t in integrand_cache.keys():
					i = integrand_cache[t]
				else:
					z = segment(t)
					i = (phi(z)*psi(z) * df(z)/f(z))/(2j*pi) * segment.dzdt(t)
					integrand_cache[t] = i
				return i

			# integrate real part
			integrand_real = lambda t: np.real(integrand(t))
			result_real = scipy.integrate.quad(integrand_real, 0, 1, full_output=1, epsabs=absTol, epsrel=relTol)
			I_real, abserr_real, infodict_real = result_real[:3]

			# integrate imaginary part			
			integrand_imag = lambda t: np.imag(integrand(t))
			result_imag = scipy.integrate.quad(integrand_imag, 0, 1, full_output=1, epsabs=absTol, epsrel=relTol)
			I_imag, abserr_imag, infodict_imag = result_imag[:3]

			I   += I_real + 1j*I_imag
			err += abserr_real + 1j*abserr_imag

		return I, abs(err)


class RootError(RuntimeError):
	pass

def count_enclosed_roots(C, f, df=None, NintAbsTol=0.07, integerTol=0.2, divMax=20, method='quad', verbose=False):
	r"""
	For a function of one complex variable, f(z), which is analytic in and within the contour C,
	return the number of zeros (counting multiplicities) within the contour calculated, using 
	Cauchy's argument principle, as
	
	.. math::

		\frac{1}{2i\pi} \oint_C \frac{f'(z)}{f(z)} dz.

	If df(z), the derivative of f(z), is provided then the above integral is computed directly.
	Otherwise the derivative is approximated using a finite difference approximation implemented
	in Numdifftools <https://pypi.python.org/pypi/Numdifftools>`_.

	The number of points on each segment of the contour C at which f(z) and df(z) are sampled 
	starts at 2+1 and at the k-th iteration the number of points is 2**k+1.  At each iteration 
	the above integral is calculated using `SciPy's implementation of the Romberg method <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html>`_.
	The routine exits if the difference between successive iterations is < integerTol.

	The number of roots is then the closest integer to the final value of the integral
	and the result is only accepted if the final value of the integral is within integerTol
	of the closest integer.
	
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
		The evaluation of the Cauchy integral will be accepted if the difference between successive
		iterations is < integerTol and the value of the integral is within integerTol of the 
		closest integer.  Since the Cauchy integral must be an integer it is only necessary to
		distinguish which integer the integral is converging towards.  For this
		reason the integerTol can be set fairly large.
	divMax : int, optional
		The maximum number of divisions before the Romberg integration
		routine of a path exits.

	Returns
	-------
	int
		The number of zeros of f (counting multiplicities) which lie within the contour
	
	References
	----------
	[DL] "A Numerical Method for Locating the Zeros of an Analytic function", 
		L.M.Delves, J.N.Lyness, Mathematics of Computation (1967), Vol.21, Issue 100
	"""
	if verbose:
		print('Computing number of roots within', C)

	with warnings.catch_warnings():
		# ignore warnings and catch if I is NaN later
		warnings.simplefilter("ignore")
		I, err = prod(C, f, df, absTol=NintAbsTol, relTol=0, divMax=divMax, method=method, verbose=verbose)

	if np.isnan(I):
		raise RootError("Result of integral is an invalid value.  Most likely because of a divide by zero error.")

	elif abs(int(round(I.real)) - I.real) < integerTol and abs(I.imag) < integerTol:
		# integral is sufficiently close to an integer
		numberOfZeros = int(round(I.real))
		return numberOfZeros

	else:
		raise RootError("The number of enclosed roots has not converged to an integer")

