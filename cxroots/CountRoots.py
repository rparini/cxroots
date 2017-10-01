from __future__ import division
import numpy as np
from numpy import inf, pi
import scipy.integrate
import scipy
import warnings
import numdifftools.fornberg as ndf

def prod(C, f, df=None, phi=lambda z:1, psi=lambda z:1, absTol=1e-12, relTol=1e-12, divMax=10, method='quad'):
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

	if approx_df or method == 'romb':
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

		return I[-1], abs(I[-2] - I[-1])

	elif method == 'quad':
		I, err = 0, 0
		for segment in C.segments:
			def integrand(t):
				z = segment(t)
				return (phi(z)*psi(z) * df(z)/f(z))/(2j*pi) * segment.dzdt(t)

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

def count_enclosed_roots(C, f, df=None, integerTol=0.25, integrandUpperBound=1e3, divMax=20, absTol=1e-4):
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
	
	If the contour is too close to a root then the integral will take a very long time to
	converge and it is generally more efficient to instead choose a different contour.
	For this reason the integration will be abandoned and a RuntimError raised if at any 
	point abs(df(z)/f(z)) > integrandUpperBound since, according to [DL], the value of 
	abs(df(z)/f(z)) is of the order of 1/(distance to nearest root).  If df is being 
	approximated then the routine will wait for the maximum value of abs(df(z)/f(z)) on
	the contour to settle down a little before considering if the routine should exit,
	since an inaccurate df(z) can cause abs(df(z)/f(z)) to be erroneously large.

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
	integrandUpperBound : float, optional
		The maximum allowed value of abs(df(z)/f(z)).  If abs(df(z)/f(z)) exceeds this 
		value then a RuntimeError is raised.  If integrandUpperBound is too large then 
		integrals may take a very long time to converge and it is generally be more 
		efficient to allow the rootfinding procedure to instead choose another contour 
		then spend time evaluating the integral along a contour very close to a root.
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
	N = 1
	I = []
	integrandMax = []

	minimum_iterations = 2
	approx_df = False
	if df is None:
		approx_df = True
		minimum_iterations = 5

	# XXX: define err as the difference between successive iterations of the Romberg
	# 	   method for the same number of points?
	while len(I) < minimum_iterations or abs(I[-2] - I[-1]) > integerTol or abs(int(round(I[-1].real)) - I[-1].real) > integerTol or abs(I[-1].imag) > integerTol or int(round(I[-1].real)) < 0:
		N = 2*N
		t = np.linspace(0,1,N+1)
		k = int(np.log2(len(t)-1))
		dt = t[1]-t[0]

		if k > divMax:
			break

		# compute/retrieve function evaluations
		fVal = np.array([segment.trapValues(f,k) for segment in C.segments])

		if approx_df:
			### approximate df/dz with finite difference, see: numdifftools.fornberg
			# interior stencil size = 2*m + 1
			# boundary stencil size = 2*m + 2
			m = 1
			dfdt = [ndf.fd_derivative(fx, t, n=1, m=m) for fx in fVal]
			dfVal = [dfdt[i]/segment.dzdt(t) for i, segment in enumerate(C.segments)]

		else:
			dfVal = np.array([segment.trapValues(df,k) for segment in C.segments])


		with warnings.catch_warnings():
			warnings.simplefilter("ignore")

			# discard the integration if it is too close to the contour
			if not approx_df:
				# if no approximation to df is being made then immediately exit if the 
				# integrand is too large
				if np.any(np.abs(dfVal/fVal) > integrandUpperBound):
					raise RootError("The absolute value of the integrand |dfVal/fVal| > integrandUpperBound which indicates that the contour is too close to zero of f(z)")

			else:
				# if df is being approximated then the integrand might be artificially
				# large so wait until the maximum value has settled a little
				integrandMax.append(np.max(np.abs(dfVal/fVal)))
				if len(integrandMax) > 1 and abs(integrandMax[-2] - integrandMax[-1]) < 0.1*integrandUpperBound:
					if np.any(np.abs(dfVal/fVal) > integrandUpperBound):
						raise RootError("The absolute value of the integrand |dfVal/fVal| > integrandUpperBound which indicates that the contour is too close to zero of f(z)")

			segment_integrand = [dfVal[i]/fVal[i]*segment.dzdt(t) for i, segment in enumerate(C.segments)]
			segment_integral  = [scipy.integrate.romb(integrand, dx=dt)/(2j*pi) for integrand in segment_integrand]
			I.append(sum(segment_integral))

			# print('k', k, 'I[-1]', I[-1])
			if k>1 and abs(I[-2].real - I[-1].real) < absTol:
				raise RootError("The number of enclosed roots has not converged to an integer")

			if np.isnan(I[-1]):
				raise RootError("Result of integral is an invalid value.  Most likely because of a divide by zero error.")



	numberOfZeros = int(round(I[-1].real))
	return numberOfZeros
