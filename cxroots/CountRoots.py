from __future__ import division
import warnings

import numpy as np
from numpy import inf, pi
import scipy.integrate
import scipy.misc
import numdifftools.fornberg as ndf
import numdifftools

def prod(C, f, df=None, phi=None, psi=None, absTol=1e-12, relTol=1e-12, divMin=3,
	divMax=15, m=2, intMethod='quad', integerTol=inf, verbose=False, callback=None):
	r"""
	Compute the symmetric bilinear form used in (1.12) of [KB]_.

	.. math::

		<\phi,\psi> = \frac{1}{2\pi i} \oint_C \phi(z) \psi(z) \frac{f'(z)}{f(z)} dz.

	Parameters
	----------
	C : :class:`Contour <cxroots.Contour.Contour>`
		A contour in the complex plane for.  No roots or poles of f
		should lie on C.
	f : function
		Function of a single variable f(x)
	df : function, optional
		Function of a single variable, df(x), providing the derivative
		of the function f(x) at the point x.  If not provided then df is
		approximated using a finite difference method.
	phi : function, optional
		Function of a single variable phi(x).  If not provided then
		phi(z)=1.
	psi : function, optional
		Function of a single variable psi(x).  If not provided then
		psi(z)=1.
	absTol : float, optional
		Absolute error tolerance for integration.
	relTol : float, optional
		Relative error tolerance for integration.
	divMin : int, optional
		Only used if intMethod='romb'. Minimum number of divisions before
		the Romberg integration routine is allowed to exit.
	divMax : int, optional
		Only used if intMethod='romb'.  The maximum number of divisions
		before the Romberg integration routine of a path exits.
	m : int, optional
		Only used if df=None and intMethod='quad'.  Must be even.  The
		argument order=m is passed to numdifftools.Derivative and is the
		order of the error term in the Taylor approximation.
	intMethod : {'quad', 'romb'}, optional
		If 'quad' then scipy.integrate.quad is used to perform the
		integral.  If 'romb' then Romberg integraion, using
		scipy.integrate.romb, is performed instead.
	integerTol : float, optional
		Only used when intMethod is 'romb'.  The integration routine will
		not exit unless the result is within integerTol of an integer.
		This is useful when computing the number of roots in a contour,
		which must be an integer.  By default integerTol is inf.
	verbose : bool, optional
		If True runtime information will be printed.  False be default.
	callback : function, optional
		Only used when intMethod is 'romb'.  A function that at each
		step in the iteration is passed the current approximation for
		the integral, the estimated error of that approximation and the
		number of iterations.  If the return of callback evaluates to
		True then the integration will end.

	Returns
	-------
	complex
		The value of the integral <phi, psi>.
	float
		An estimate of the error for the integration.

	References
	----------
	.. [KB] "Computing the zeros of analytic functions" by Peter Kravanja,
		Marc Van Barel, Springer 2000
	"""
	if intMethod == 'romb':
		N = 1
		k = 0
		I = []

		while k < divMax and (len(I) < divMin
			or (abs(I[-2] - I[-1]) > absTol and abs(I[-2] - I[-1]) > relTol*abs(I[-1]))
			or (abs(I[-3] - I[-2]) > absTol and abs(I[-3] - I[-2]) > relTol*abs(I[-2]))
			or abs(int(round(I[-1].real)) - I[-1].real) > integerTol
			or abs(I[-1].imag) > integerTol):
			N = 2*N
			t = np.linspace(0,1,N+1)
			k += 1
			dt = t[1]-t[0]

			integrals = []
			for segment in C.segments:
				# compute/retrieve function evaluations
				fVal = segment.trap_values(f,k)

				if df is None:
					# approximate df/dz with finite difference
					dfdt = np.gradient(fVal, dt)
					dfVal = dfdt/segment.dzdt(t)
				else:
					dfVal = segment.trap_values(df,k)

				segment_integrand = dfVal/fVal*segment.dzdt(t)
				if phi is not None:
					segment_integrand = segment.trap_values(phi,k)*segment_integrand
				if psi is not None:
					segment_integrand = segment.trap_values(psi,k)*segment_integrand

				segment_integral = scipy.integrate.romb(segment_integrand, dx=dt, axis=-1)/(2j*pi)
				integrals.append(segment_integral)

			I.append(sum(integrals))

			if verbose:
				if k > 1:
					print(k, 'I', I[-1], 'err', I[-2] - I[-1])
				else:
					print(k, 'I', I[-1])

			if callback is not None:
				err = abs(I[-2] - I[-1]) if k > 1 else None
				if callback(I[-1], err, k):
					break

		return I[-1], abs(I[-2] - I[-1])

	elif intMethod == 'quad':
		if df is None:
			df = numdifftools.Derivative(f, order=m)
			# df = lambda z: scipy.misc.derivative(f, z, dx=1e-8, n=1, order=3)

			### Too slow
			# ndf.derivative returns an array [f, f', f'', ...]
			# df = np.vectorize(lambda z: ndf.derivative(f, z, n=1)[1])

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

	else:
		raise ValueError("intMethod must be either 'romb' or 'quad'")

class RootError(RuntimeError):
	pass

def count_roots(C, f, df=None, NIntAbsTol=0.07, integerTol=0.1, divMin=3,
	divMax=15, m=2, intMethod='quad', verbose=False):
	r"""
	For a function of one complex variable, f(z), which is analytic in
	and within the contour C, return the number of zeros (counting
	multiplicities) within the contour, N, using Cauchy's argument
	principle,

	.. math::

		N = \frac{1}{2i\pi} \oint_C \frac{f'(z)}{f(z)} dz.

	If df(z), the derivative of f(z), is provided then the above
	integral is computed directly.  Otherwise the derivative is
	approximated using a finite difference method.

	The number of roots is taken to be the closest integer to the
	computed value of the integral and the result is only accepted
	if the integral is within integerTol of the closest integer.

	Parameters
	----------
	C : :class:`Contour <cxroots.Contour.Contour>`
		The contour which encloses the roots of f(z) that are to be
		counted.
	f : function
		Function of a single variable f(z).
	df : function, optional
		Function of a single complex variable, df(z), providing the
		derivative of the function f(z) at the point z.  If not
		provided, df will be approximated using a finite difference
		method.
	NIntAbsTol : float, optional
		Required absolute error tolerance for the contour integration.
		Since the Cauchy integral must be an integer it is only
		necessary to distinguish which integer the integral is
		converging towards.  Therefore, NIntAbsTol can be fairly large.
	integerTol : float, optional
		The evaluation of the Cauchy integral will be accepted if its
		value is within integerTol of the closest integer.
	divMin : int, optional
		Only used if intMethod='romb'. Minimum number of divisions
		before the Romberg integration routine is allowed to exit.
	divMax : int, optional
		Only used if intMethod='romb'.  The maximum number of divisions
		before the Romberg integration routine of a path exits.
	m : int, optional
		Only used if df=None and intMethod='quad'.  The argument order=m
		is passed to numdifftools.Derivative and is the order of the
		error term in the Taylor approximation.  m must be even.
	intMethod : {'quad', 'romb'}, optional
		If 'quad' then scipy.integrate.quad is used to perform the
		integral.  If 'romb' then Romberg integraion, using
		scipy.integrate.romb, is performed instead.
	verbose : bool, optional
		If True certain messages regarding the integration will be
		printed.

	Returns
	-------
	int
		The number of zeros of f (counting multiplicities) which lie
		within the contour C.
	"""
	if verbose:
		print('Computing number of roots within', C)

	with warnings.catch_warnings():
		# ignore warnings and catch if I is NaN later
		warnings.simplefilter("ignore")
		I, err = prod(C, f, df, absTol=NIntAbsTol, relTol=0, divMin=divMin,
			divMax=divMax, m=m, intMethod=intMethod, verbose=verbose, integerTol=integerTol)

	if intMethod == 'romb':
		C._numberOfDivisionsForN = int(np.log2(len(C.segments[0]._trapValuesCache[f])-1))

	if np.isnan(I):
		raise RootError("""Result of integral is an invalid value.
						   Most likely because of a divide by zero error.""")

	elif abs(int(round(I.real)) - I.real) < integerTol and abs(I.imag) < integerTol:
		# integral is sufficiently close to an integer
		numberOfZeros = int(round(I.real))
		return numberOfZeros

	else:
		raise RootError("The number of enclosed roots has not converged to an integer")

