from __future__ import division
import functools

import numpy as np
import scipy.linalg

from .CountRoots import prod

def approximate_roots(C, N, f, df=None, absTol=1e-12, relTol=1e-12,
	errStop=1e-10, divMin=3, divMax=15, m=2, rootTol=1e-8,
	intMethod='quad', callback=None, verbose=False):
	"""
	Approximate the roots and multiplcities of the function f within the
	contour C using the method of [KB]_.  The multiplicites are computed
	using eq. (21) in [SLV]_.

	Parameters
	----------
	C : :class:`~<cxroots.Contour.Contour>`
		The contour which encloses the roots of f the user wishes to find.
	N : int
		The number of roots (counting multiplicties) of f within C.
		This is the result of calling :meth:`~cxroots.Contour.Contour.count_roots`.
	f : function
		The function for which the roots are sought.  Must be a function
		of a single complex variable, z, which is analytic within C and
		has no poles or roots on the C.
	df : function, optional
		A function of a single complex variable which is the derivative
		of the function f(z). If df is not given then it will be
		approximated with a finite difference formula.
	absTol : float, optional
		Absolute error tolerance for integration.
	relTol : float, optional
		Relative error tolerance for integration.
	errStop : float, optional
		The number of distinct roots within a contour, n, is determined
		by checking if all the elements of a list of contour integrals
		involving formal orthogonal polynomials are sufficently close to
		zero, ie. that the absolute value of each element is < errStop.
		If errStop is too large/small then n may be smaller/larger than
		it actually is.
	divMin : int, optional
		If the Romberg integration method is used then divMin is the
		minimum number of divisions before the Romberg integration
		routine is allowed to exit.
	divMax : int, optional
		If the Romberg integration method is used then divMax is the
		maximum number of divisions before the Romberg integration
		routine exits.
	m : int, optional
		Only used if df=None and method='quad'.  The argument order=m is
		passed to numdifftools.Derivative and is the order of the error
		term in the Taylor approximation.  m must be even.
	rootTol : float, optional
		If any roots are within rootTol of one another then they will be
		treated as duplicates and removed.  This helps to alleviate the
		problem of errStop being too small.
	intMethod : {'quad', 'romb'}, optional
		If 'quad' then :func:`scipy.integrate.quad` is used to perform
		integration.  If 'romb' then Romberg integraion is performed
		instead.
	callback : function, optional
		Only used if intMethod is 'romb'.  Passed to :func:`~<cxroots.CountRoots.prod>`.
	verbose : bool, optional
		If True certain information regarding the rootfinding process
		will be printed.

	Returns
	-------
	tuple of complex
		The distinct roots of f within the contour C.
	tuple of float
		The corresponding multiplicites of the roots within C.  Should
		be integers but will not be automatically rounded here.

	References
	----------
	.. [KB] P. Kravanja and M. Van Barel. "Computing the Zeros of
		Anayltic Functions". Springer (2000)
	.. [SLV] E. Strakova, D. Lukas, P. Vodstrcil. "Finding Zeros of
		Analytic Functions and Local Eigenvalue Analysis Using Contour
		Integral Method in Examples". Mathematical Analysis and Numerical
		Mathematics, Vol. 15, 2, (2017)
	"""
	if verbose:
		print('Approximating roots in: ' + str(C))
		print('The number of roots, counting multiplcities, within this contour is', N)

	if N == 0:
		return (), ()

	product = functools.partial(prod, C, f, df,
		absTol=absTol, relTol=relTol, divMin=divMin, divMax=divMax,
		m=m, intMethod=intMethod, verbose=verbose, callback=callback)

	s = [N, product(lambda z: z)[0]]	# ordinary moments
	mu = s[1]/N
	phiZeros = [[],[mu]]

	def phi(i):
		if len(phiZeros[i]) == 0:
			return lambda z: np.ones_like(z)
		else:
			coeff = np.poly(phiZeros[i])
			return lambda z: np.polyval(coeff, z)

	# initialize G_{pq} = <phi_p, phi_q>
	G = np.zeros((N,N), dtype=np.complex128)
	G[0,0] = N # = <phi_0, phi_0> = <1,1>

	# initialize G1_{pq} = <phi_p, phi_1 phi_q>
	G1 = np.zeros((N,N), dtype=np.complex128)
	G1[0,0] = 0 # = <phi_0, phi_1 phi_0> = <1, z-mu> = s1-mu*N = 0

	r, t = 1, 0
	while r+t<N:
		### define FOP of degree r+t+1
		p = r+t
		G[p, 0:p+1] = [product(phi(p), phi(q))[0] for q in range(r+t+1)]
		G[0:p+1, p] = G[p, 0:p+1] # G is symmetric
		if verbose: print('G ', G[:p+1,:p+1])

		G1[p, 0:p+1] = [product(phi(p), lambda z: phi(1)(z)*phi(q)(z))[0] for q in range(r+t+1)]
		G1[0:p+1, p] = G1[p, 0:p+1] # G1 is symmetric
		if verbose: print('G1', G1[:p+1,:p+1])

		"""
		If any of the zeros of the FOP are outside of the interior
		of the contour then we assume that they are 'arbitary' and
		instead define the FOP as an inner polynomial. [KB]
		"""
		polyRoots = scipy.linalg.eig(G1[:p+1,:p+1], G[:p+1,:p+1])[0]+mu
		if np.all([C.contains(z) for z in polyRoots]):
			r, t = r+t+1, 0
			phiZeros.append(polyRoots)

			if verbose: print('Regular poly', r+t, 'roots:', phiZeros[-1])

			# is the number of distinct roots, n=r?
			phiFuncLast = phi(-1)
			for j in range(N-r):
				ip, err = product(lambda z: phiFuncLast(z)*(z-mu)**j, phiFuncLast)

				if verbose: print(j, 'of', N-r, 'err', err, 'abs(ip)', abs(ip))
				if abs(ip) > errStop:
					# n != r so carry on
					if verbose: print('n !=', r)
					break
			else:
				# the for loop did not break
				if verbose: print('n =', r)
				break

		else:
			# define an inner polynomial as phi_{r+t+1} = phi_{t+1} phi_{r}
			t += 1
			phiZeros.append(np.append(phiZeros[t],phiZeros[r]))
			if verbose: print('Inner poly', r+t, 'roots:', phiZeros[-1])

	roots = np.array(phiZeros[-1])

	if verbose:
		print('Computed Roots:')
		print(roots)

	# remove any roots which are not distinct
	rootsToRemove = []
	for i, root in enumerate(roots):
		if len(roots[i+1:]) > 0 and np.any(np.abs(root-roots[i+1:]) < rootTol):
			rootsToRemove.append(i)
	roots = np.delete(roots, rootsToRemove)
	n = len(roots)

	### compute the multiplicities, eq. (1.19) in [KB]
	# V = np.column_stack([roots**i for i in range(n)])
	# if verbose and n > 2: print('Computing ordinary moments')
	# s += [product(lambda z: z**p)[0] for p in range(2, n)]
	# multiplicities = np.dot(s[:n], np.linalg.inv(V))

	### compute the multiplicities, eq. (21) in [SLV]
	V = np.array([[phi(j)(root) for root in roots] for j in range(n)])
	multiplicities = np.dot(np.linalg.inv(V), G[:n,0])

	### The method used in the vandermonde module doesn't seem significantly
	### better than np.dot(s, np.linalg.inv(V)).  Especially since we know
	### the result must be an integer anyway.
	# import vandermonde
	# multiplicities_vandermonde = vandermonde.solve_transpose(np.array(roots), np.array(s))

	### Note that n = rank(H_N) is not used since calculating the
	### rank of a matrix of floats can be quite unstable
	# s_func = lambda p: prod(C, f, df, lambda z: z**p)[0]
	# HN = np.fromfunction(np.vectorize(lambda p,q: s_func(p+q)), shape=(N,N))
	# print('n?', np.linalg.matrix_rank(HN, tol=1e-10))

	if verbose:
		print('Approximations for roots:\n', roots)
		print('Approximations for multiplicities:\n', multiplicities)

	return tuple(roots), tuple(multiplicities)

