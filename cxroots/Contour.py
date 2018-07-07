"""
References
----------
[DSZ] "Locating all the Zeros of an Analytic Function in one Complex Variable"
	M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2
[BVP] Gakhov, F. D. "Boundary value problems", section 12 (2014), Elsevier.
[DL] "A Numerical Method for Locating the Zeros of an Analytic function", 
	L.M.Delves, J.N.Lyness, Mathematics of Computation (1967), Vol.21, Issue 100
[KB] "Computing the Zeros of Anayltic Functions", Peter Kravanja, Marc Van Barel,
	Springer (2000)
"""

from __future__ import division
import numpy as np
import scipy.integrate
import scipy.linalg

from .CountRoots import count_roots, prod
from .RootFinder import findRoots, MultiplicityError
from .DemoRootFinder import demo_findRoots
from .Misc import doc_tab_to_space, docstrings, NumberOfRootsChanged

class Contour(object):
	def __init__(self, segments):
		self.segments = np.array(segments, dtype=object)

	def __call__(self, t):
		t = np.array(t)
		N = len(self.segments)
		segmentIndex = np.array(N*t, dtype=int)
		segmentIndex = np.mod(segmentIndex, N)
		segment = self.segments[segmentIndex]

		if hasattr(segmentIndex, '__iter__'):
			return np.array([self.segments[i](N*t[ti]%1) for ti, i in enumerate(segmentIndex)])
		else:
			return self.segments[segmentIndex](N*t%1)

	def plot(self, *args, **kwargs):
		for segment in self.segments:
			segment.plot(*args, **kwargs)

	def sizePlot(self):
		import matplotlib.pyplot as plt
		t = np.linspace(0,1,1000)
		z = self(t)
		xpad = (max(np.real(z))-min(np.real(z)))*0.1
		ypad = (max(np.imag(z))-min(np.imag(z)))*0.1

		xmin = min(np.real(z))-xpad
		xmax = max(np.real(z))+xpad
		ymin = min(np.imag(z))-ypad
		ymax = max(np.imag(z))+ypad
		plt.xlim([xmin, xmax])
		plt.ylim([ymin, ymax])

	def show(self, saveFile=None, *args, **kwargs):
		""" 
		Shows the contour as a 2D plot in the complex plane.

		Parameters
		==========
		saveFile : str (optional)
			If given then the plot will be saved to disk with name 'saveFile' instead of being shown.
		"""
		import matplotlib.pyplot as plt
		self.sizePlot()
		self.plot(*args, **kwargs)

		if saveFile is not None:
			plt.savefig(saveFile)
			plt.close()
		else:
			plt.show()

	def subdivisions(self, axis='alternating'):
		""" 
		A generator of possible subdivisions of the contour, starting with an equal subdivision. 
		
		Parameters
		----------
		axis : str, 'alternating' or any element of self.axisName which is defined in a Contour subclass
			The axis along which the line subdividing the contour is a constant (eg. subdividing a circle
			along the radial axis will give an outer annulus and an inner circle).  If alternating then
			the dividing axis will always be different to the dividing axis used to create the contour
			which is now being divided.

		Returns
		-------
		Generator of possible subdivisions
		"""
		if axis == 'alternating':
			# if the box to be subdivided was itself created by subdivision then alternate the axis of subdivision
			if hasattr(self,'_createdBySubdivisionAxis'):
				axis = (self._createdBySubdivisionAxis + 1)%len(self.axisName)
			else:
				axis = 0

		for divisionFactor in divisionFactorGen():
			yield self.subdivide(axis, divisionFactor)

	def distance(self, P):
		"""
		Get the distance from the point P in the complex plane to the nearest point on the contour.
		"""
		return min(segment.distance(P) for segment in self.segments)

	def integrate(self, f, absTol=0, relTol=1e-12, rombergDivMax=10, method='quad', verbose=False):
		""" Integrate around the contour, same arguments the integrate method for ComplexPath """
		return sum([segment.integrate(f, absTol, relTol, rombergDivMax, method, verbose) for segment in self.segments])

	def count_roots(self, *args, **kwargs):
		return count_roots(self, *args, **kwargs)

	def approximate_roots(self, f, df=None, absTol=1e-12, relTol=1e-12, NAbsTol=0.07, integerTol=0.1, errStop=1e-8, 
		divMin=5, divMax=10, m=2, rootTol=1e-8, intMethod='quad', verbose=False, M=None):
		if verbose:
			print('Approximating roots in: ' + str(self))

		if not hasattr(self, '_numberOfRoots'):
			self._numberOfRoots = self.count_roots(f, df, NAbsTol, integerTol, divMin, divMax, m, intMethod, verbose)
		N = self._numberOfRoots

		if verbose:
			print(N, 'Roots in', str(self))

		if N == 0:
			return (), ()

		if intMethod == 'romb':
			# Check to see if the number of roots has changed after new values of f have been sampled
			vals = self.segments[0]._trapValuesCache[f]
			self._numberOfDivisionsForN = int(np.log2(len(vals)-1))

			def callback(I):
				if len(I) > self._numberOfDivisionsForN:
					if verbose:
						print('--- Checking N using the newly sampled values of f ---')
					new_N = self.count_roots(f, df, NAbsTol, integerTol, len(I), divMax, m, intMethod, verbose)
					if verbose:
						print('------------------------------------------------------')

					# update numberOfDivisionsForN
					vals = self.segments[0]._trapValuesCache[f]
					self._numberOfDivisionsForN = int(np.log2(len(vals)-1))

					if new_N != N:
						if verbose:			
							print('N has been recalculated using more samples of f')
						self._numberOfRoots = new_N
						raise NumberOfRootsChanged
		else:
			callback = None

		try:
			mu = prod(self, f, df, lambda z: z, None, absTol, relTol, divMin, divMax, m, intMethod, verbose, callback)[0]/N
			phiZeros = [[],[mu]]

			def phiFunc(i):
				if len(phiZeros[i]) == 0:
					return lambda z: np.ones_like(z)
				else:
					coeff = np.poly(phiZeros[i])
					return lambda z: np.polyval(coeff, z)
			
			# initialize G_{pq} = <phi_p, phi_q>
			G = np.zeros((N,N), dtype=np.complex128)
			G[0,0] = N # = <phi0, phi0> = <1,1>

			# initialize G1_{pq} = <phi_p, phi_1 phi_q>
			G1 = np.zeros((N,N), dtype=np.complex128)
			phi1 = phiFunc(1)
			ip, err = prod(self, f, df, phiFunc(0), lambda z: phi1(z)*phiFunc(0)(z), absTol, relTol, divMin, divMax, m, intMethod, verbose, callback)
			G1[0,0] = ip

			take_regular = True

			r, t = 1, 0
			while r+t<N:
				# define the next FOP of degree r+t+1
				k = r+t+1

				# Add new values to G
				p = r+t
				G[p, 0:p+1] = [prod(self, f, df, phiFunc(p), phiFunc(q), absTol, relTol, divMin, divMax, m, intMethod, verbose, callback)[0] for q in range(r+t+1)]
				G[0:p+1, p] = G[p, 0:p+1] # G is symmetric

				# Add new values to G1
				G1[p, 0:p+1] = [prod(self, f, df, phiFunc(p), lambda z: phi1(z)*phiFunc(q)(z), absTol, relTol, divMin, divMax, m, intMethod, verbose, callback)[0] for q in range(r+t+1)]
				G1[0:p+1, p] = G1[p, 0:p+1] # G1 is symmetric

				if verbose:
					print('G ', G[:p+1,:p+1])
					print('G1', G1[:p+1,:p+1])

				# The regular FOP only exists if H is non-singular.
				# An alternate citeration given by [KB] is to proceed as if it is regular and
				# then compute its zeros.  If any are arbitary or infinite then this
				# polynomial should instead be defined as an inner polynomial.
				# Here, an inner polynomial is defined if any of the computed
				# roots are outside of the interior of the contour.
				polyRoots = scipy.linalg.eig(G1[:p+1,:p+1], G[:p+1,:p+1])[0]+mu
				if np.all([self.contains(z) for z in polyRoots]):
					# define a regular polynomial
					phiZeros.append(polyRoots)
					r, t = r+t+1, 0

					if verbose:
						print('Regular poly', r+t, 'roots:', phiZeros[-1])

					# if any of these elements are not small then continue
					allSmall = True
					phiFuncLast = phiFunc(-1)
					for j in range(N-r):
						ip, err = prod(self, f, df, lambda z: phiFuncLast(z)*(z-mu)**j, phiFuncLast, absTol, relTol, divMin, divMax, m, intMethod, verbose, callback)

						# if not small then carry on
						if verbose:
							print(j, 'of', N-r, 'stop?', abs(ip) + err)
						### XXX: Use the 'maxpsum' estimate for precision loss in [KB]?
						if abs(ip) + err > errStop:
							allSmall = False
							break

					if allSmall:
						# all the roots have been found
						break

				else:
					t += 1

					# define an inner polynomial phi_{r+t+1} = phi_{t+1} phi_{r}
					phiZeros.append(np.append(phiZeros[t],phiZeros[r]))

					if verbose:
						print('Inner poly', r+t, 'roots:', phiZeros[-1])

			roots = np.array(phiZeros[-1])

			if verbose:
				print('Roots:')
				print(roots)

			# remove any roots which are not distinct
			removeList = []
			for i, root in enumerate(roots):
				if len(roots[i+1:]) > 0 and np.any(np.abs(root-roots[i+1:]) < rootTol):
					removeList.append(i)

			roots = np.delete(roots, removeList)

			if verbose:
				print('Post-removed roots:')
				print(roots)

			n = len(roots) # number of distinct roots

			# compute the multiplicities, eq. (1.19) in [KB]
			V = np.column_stack([roots**i for i in range(n)])
			from time import time
			if verbose:
				print('Computing ordinary moments')
			s = [N] 	# = s0
			s += [prod(self, f, df, lambda z: z**p, None, absTol, relTol, divMin, divMax, m, intMethod, verbose, callback)[0] for p in range(1, n)] 	# ordinary moments
			multiplicities = np.dot(s, np.linalg.inv(V))

			### The method used in the vandermonde module doesn't seem significantly
			### better than np.dot(s, np.linalg.inv(V)).  Especially since we know
			### the result must be an integer anyway.
			# import vandermonde
			# multiplicities = vandermonde.solve_transpose(np.array(roots), np.array(s))

			### Note that n = rank(H_N) is not used since calculating the
			### rank of a matrix of floats appears to be quite unstable
			# s_func = lambda p: prod(self, f, df, lambda z: z**p)[0]
			# HN = np.fromfunction(np.vectorize(lambda p,q: s_func(p+q)), shape=(N,N))
			# print('n?', np.linalg.matrix_rank(HN, tol=1e-10))

			if verbose:
				print('Computed multiplicities:')
				print(multiplicities)

			# round multiplicities
			rounded_multiplicities = np.round(multiplicities)
			rounded_multiplicities = np.array([int(m.real) for m in rounded_multiplicities])
			if np.all(np.abs(rounded_multiplicities - np.real(multiplicities)) < integerTol) and np.all(np.abs(np.imag(multiplicities)) < integerTol):
				multiplicities = rounded_multiplicities
			else:
				# multiplicities are not sufficiently close to integers
				raise MultiplicityError("Some multiplicities are not integers:", multiplicities)

			# remove any roots with multiplicity zero
			zeroArgs = np.where(multiplicities == 0)
			multiplicities = np.delete(multiplicities, zeroArgs)
			roots = np.delete(roots, zeroArgs)

			if verbose:
				print('Computed roots:')
				print(roots)
				print('Final multiplicities:')
				print(multiplicities)

			return tuple(roots), tuple(multiplicities)

		except NumberOfRootsChanged:
			# The total number of roots changed so repeat the rootfinding approximation
			if M is not None and self._numberOfRoots > M:
				# The number of roots in this contour is bigger than the allowed value
				raise NumberOfRootsChanged
			return self.approximate_roots(f, df, absTol, relTol, NAbsTol, integerTol, errStop, divMin, divMax, m, rootTol, intMethod, verbose)

	def roots(self, f, df=None, **kwargs):
		return findRoots(self, f, df, **kwargs)

	def demo_roots(self, *args, **kwargs):
		"""
		An animated demonstration of the root finding process using matplotlib.
		Takes all the parameters of :func:`Contour.roots <cxroots.Contours.Contour.roots>` as well as:

		Parameters
		----------
		automaticAnim : bool, optional
			If False (default) then press SPACE to step the animation forward
			If True then the animation will play automatically until all the 
			roots have been found.
		saveFile : str, optional
			If given then the animation will be saved to disk with filename 
			equal to saveFile instead of being shown.
		returnAnim : bool, optional
			If True then the matplotlib animation object will be returned 
			instead of being shown.  Defaults to False.
		"""
		return demo_findRoots(self, *args, **kwargs)

	def show_roots(self, *args, **kwargs):
		roots = self.roots(*args, **kwargs)
		roots.show()

	def print_roots(self, *args, **kwargs):
		roots = self.roots(*args, **kwargs)
		print(roots)

# Reuse docs for roots
try:
	Contour.roots.__doc__ = docstrings.delete_params_s(findRoots.__doc__, ['originalContour'])
except AttributeError:
	# for Python 2.7
	Contour.roots.__func__.__doc__ = docstrings.delete_params_s(findRoots.__doc__, ['originalContour'])


def divisionFactorGen():
	"""A generator for divisionFactors"""
	yield 0.3	# being off-center is a better first choice for certain problems
	
	x = 0.5
	yield x
	for power in [1e1, 1e2, 1e3]:
		for diff in np.linspace(0, 0.5, int(1+power/2))[1:-1]:
			yield x + diff
			yield x - diff
