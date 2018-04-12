from __future__ import division
import numpy as np
from numpy import inf, pi
import scipy.integrate
import scipy.misc
import warnings

from .CxDerivative import CxDeriv

def prod(C, f, df=None, phi=None, psi=None, absTol=1e-12, relTol=1e-12, divMin=5, divMax=10, m=2, method='quad', verbose=False):
	r"""
	Compute the symmetric bilinear form used in (1.12) of [KB]

	.. math::

		<\phi,\psi> = \frac{1}{2i\pi} \oint_C \phi(z) \psi(z) \frac{f'(z)}{f(z)} dz.
	
    Parameters
    ----------
	C : Contour
		The enclosed_roots function returns the number of roots of f(z) within C
	f : function
		Function of a single variable f(x)
	df : function, optional
		Function of a single variable, df(x), providing the derivative of the function f(x) 
		at the point x.  If not provided then df is approximated using a finite difference
		method.
	phi : function, optional
		Function of a single variable phi(x).  If not provided then phi=1.
	psi : function, optional
		Function of a single variable psi(x).  If not provided then psi=1.
	absTol : float, optional
		Absolute error tolerance.
	relTol : float, optional
		Relative error tolerance.
 	divMin : int, optional
 		Minimum number of divisions before the Romberg integration routine is allowed 
 		to exit.  Only used if method='romb'.
	divMax : int, optional
		The maximum number of divisions before the Romberg integration routine of a 
		path exits.  Only used if method='romb'.
    m : int, optional
    	Only used if df=None.  If method='romb' then m defines the stencil size for the 
    	numerical differentiation of f, passed to numdifftools.fornberg.fd_derivative.
    	The stencil size is of 2*m+1 points in the interior, and 2*m+2 points for each 
    	of the 2*m boundary points.  If instead method='quad' then m must is the order of 
    	the error term in the Taylor approximation used which must be even.  The argument
    	order=m is passed to numdifftools.Derivative.
	method : {'quad', 'romb'}, optional
		If 'quad' then scipy.integrate.quad is used to perform the integral.  If 'romb'
		then Romberg integraion, using scipy.integrate.romb, is performed instead.

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

	if method == 'romb':
		import numdifftools.fornberg as ndf
		# XXX: Better way to characterise err than abs(I[-2] - I[-1])?
		while (len(I) < divMin or (abs(I[-2] - I[-1]) > absTol and abs(I[-2] - I[-1]) > relTol*abs(I[-1]))) and k < divMax:
			N = 2*N
			t = np.linspace(0,1,N+1)
			k = int(np.log2(len(t)-1))
			dt = t[1]-t[0]

			integrals = []
			for segment in C.segments:
				# compute/retrieve function evaluations
				fVal = segment.trapValues(f,k)

				if approx_df:
					### approximate df/dz with finite difference, see: numdifftools.fornberg
					used_m = m
					if 2*m+1 > len(t):
						# not enough points to accommodate stencil size
						# temporarily reduce m
						used_m = (len(t)-1)//2

					dfdt = ndf.fd_derivative(fVal, t, n=1, m=used_m)
					dfVal = dfdt/segment.dzdt(t)

				else:
					dfVal = segment.trapValues(df,k)

				segment_integrand = dfVal/fVal*segment.dzdt(t)
				if phi is not None:
					segment_integrand = segment.trapValues(phi,k)*segment_integrand
				if psi is not None:
					segment_integrand = segment.trapValues(psi,k)*segment_integrand

				segment_integral = scipy.integrate.romb(segment_integrand, dx=dt, axis=-1)/(2j*pi)
				integrals.append(segment_integral)
			
			I.append(sum(integrals))

			if verbose:
				if k > 1:
					print(k, 'I', I[-1], 'err', I[-2] - I[-1])
				else:
					print(k, 'I', I[-1])

		return I[-1], abs(I[-2] - I[-1])

	elif method == 'quad':
		if approx_df:
			import numdifftools
			df = numdifftools.Derivative(f, order=m)
			# df = lambda z: scipy.misc.derivative(f, z, dx=1e-8, n=1, order=3)
			# df = CxDeriv(f) # too slow

		I, err = 0, 0
		for segment in C.segments:
			integrand_cache = {}
			def integrand(t):
				if t in integrand_cache.keys():
					i = integrand_cache[t]
				else:
					z = segment(t)
					i = (df(z)/f(z))/(2j*pi) * segment.dzdt(t)
					if phi is not None:
						i = phi(z)*i
					if psi is not None:
						i = psi(z)*i
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

def count_enclosed_roots(C, f, df=None, NintAbsTol=0.07, integerTol=0.2, divMin=5, divMax=20, m=2, method='quad', verbose=False):
	r"""
	For a function of one complex variable, f(z), which is analytic in and within the contour C,
	return the number of zeros (counting multiplicities) within the contour calculated, using 
	Cauchy's argument principle, as
	
	.. math::

		\frac{1}{2i\pi} \oint_C \frac{f'(z)}{f(z)} dz.

	If df(z), the derivative of f(z), is provided then the above integral is computed directly.
	Otherwise the derivative is approximated using a finite difference approximation implemented
	in Numdifftools <https://pypi.python.org/pypi/Numdifftools>`_.

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
	NintAbsTol : float, optional
		Required absolute error for the integration.
	integerTol : float, optional
		The evaluation of the Cauchy integral will be accepted if the difference between successive
		iterations is < integerTol and the value of the integral is within integerTol of the 
		closest integer.  Since the Cauchy integral must be an integer it is only necessary to
		distinguish which integer the integral is converging towards.  For this
		reason the integerTol can be set fairly large.
 	divMin : int, optional
 		Minimum number of divisions before the Romberg integration routine is allowed 
 		to exit.  Only used if method='romb'.
	divMax : int, optional
		The maximum number of divisions before the Romberg integration routine of a 
		path exits.  Only used if method='romb'.
    m : int, optional
    	Only used if df=None.  If method='romb' then m defines the stencil size for the 
    	numerical differentiation of f, passed to numdifftools.fornberg.fd_derivative.
    	The stencil size is of 2*m+1 points in the interior, and 2*m+2 points for each 
    	of the 2*m boundary points.  If instead method='quad' then m must is the order of 
    	the error term in the Taylor approximation used which must be even.  The argument
    	order=m is passed to numdifftools.Derivative.
	method : {'quad', 'romb'}, optional
		If 'quad' then scipy.integrate.quad is used to perform the integral.  If 'romb'
		then Romberg integraion, using scipy.integrate.romb, is performed instead.

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
		I, err = prod(C, f, df, absTol=NintAbsTol, relTol=0, divMin=divMin, divMax=divMax, m=m, method=method, verbose=verbose)

	if np.isnan(I):
		raise RootError("Result of integral is an invalid value.  Most likely because of a divide by zero error.")

	elif abs(int(round(I.real)) - I.real) < integerTol and abs(I.imag) < integerTol:
		# integral is sufficiently close to an integer
		numberOfZeros = int(round(I.real))
		return numberOfZeros

	else:
		raise RootError("The number of enclosed roots has not converged to an integer")

