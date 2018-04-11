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
import random
import warnings
import numpy as np
import scipy.integrate
import scipy.linalg
from scipy import pi, exp, sin, log
import scipy

from .CountRoots import count_enclosed_roots, prod
from .RootFinder import findRoots, MultiplicityError
from .DemoRootFinder import demo_findRoots
from .Paths import ComplexLine, ComplexArc
from .Misc import doc_tab_to_space, docstrings

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

	def integrate(self, f, absTol=0, relTol=1e-12, rombergDivMax=10, method='quad', show=False):
		""" Integrate around the contour, same arguments the integrate method for ComplexPath """
		return sum([segment.integrate(f, absTol, relTol, rombergDivMax, method, show) for segment in self.segments])

	def count_roots(self, *args, **kwargs):
		return count_enclosed_roots(self, *args, **kwargs)

	def approximate_roots(self, f, df=None, absTol=1e-12, relTol=1e-12, NAbsTol=0.07, integerTol=0.1, errStop=1e-8, divMin=5, divMax=10, m=2, rootTol=1e-8, intMethod='quad', verbose=False):
		if hasattr(self, '_numberOfRoots'):
			N = self._numberOfRoots
		else:
			N = self.count_roots(f, df, NAbsTol, integerTol, divMin, divMax, m, intMethod, verbose)

		if N == 0:
			return (), ()

		mu = prod(self, f, df, lambda z: z, None, absTol, relTol, divMin, divMax, m, intMethod, verbose)[0]/N
		phiZeros = [[],[mu]]

		def phiFunc(i):
			if len(phiZeros[i]) == 0:
				return lambda z: np.ones_like(z)
			else:
				coeff = np.poly(phiZeros[i])
				return lambda z: np.polyval(coeff, z)
		
		if verbose:
			print('Approximating roots in: ' + str(self))
			print('mu', mu)

		# initialize G_{pq} = <phi_p, phi_q>
		G = np.zeros((N,N), dtype=np.complex128)
		G[0,0] = N # = <phi0, phi0> = <1,1>

		# initialize G1_{pq} = <phi_p, phi_1 phi_q>
		G1 = np.zeros((N,N), dtype=np.complex128)
		phi1 = phiFunc(1)
		ip, err = prod(self, f, df, phiFunc(0), lambda z: phi1(z)*phiFunc(0)(z), absTol, relTol, divMin, divMax, m, intMethod, verbose)
		G1[0,0] = ip

		take_regular = True

		r, t = 1, 0
		while r+t<N:
			# define the next FOP of degree r+t+1
			k = r+t+1

			# Add new values to G
			p = r+t
			G[p, 0:p+1] = [prod(self, f, df, phiFunc(p), phiFunc(q), absTol, relTol, divMin, divMax, m, intMethod, verbose)[0] for q in range(r+t+1)]
			G[0:p+1, p] = G[p, 0:p+1] # G is symmetric

			# Add new values to G1
			G1[p, 0:p+1] = [prod(self, f, df, phiFunc(p), lambda z: phi1(z)*phiFunc(q)(z), absTol, relTol, divMin, divMax, m, intMethod, verbose)[0] for q in range(r+t+1)]
			G1[0:p+1, p] = G1[p, 0:p+1] # G1 is symmetric

			if verbose:
				print('G', G[:p+1,:p+1])
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
					ip, err = prod(self, f, df, lambda z: phiFuncLast(z)*(z-mu)**j, phiFuncLast, absTol, relTol, divMin, divMax, m, intMethod, verbose)

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
		s = [prod(self, f, df, None, None, absTol, relTol, divMin, divMax, m, intMethod, verbose)[0]] 	# = s0
		s += [prod(self, f, df, lambda z: z**p, None, absTol, relTol, divMin, divMax, m, intMethod, verbose)[0] for p in range(1, n)] 	# ordinary moments
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
			# multiplicities are not sufficiently close to roots
			raise MultiplicityError("Some multiplicities are not integers")

		# remove any roots with multiplicity zero
		zeroArgs = np.where(multiplicities == 0)
		multiplicities = np.delete(multiplicities, zeroArgs)
		roots = np.delete(roots, zeroArgs)

		if verbose:
			print('Final roots:')
			print(roots)
			print('Final multiplicities:')
			print(multiplicities)

		return tuple(roots), tuple(multiplicities)

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



class Circle(Contour):
	"""
	A positively oriented circle in the complex plane.

	Parameters
	----------
	center : complex
		The center of the circle.
	radius : float
		The radius of the circle.
	"""
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius
		self.axisName = ['r']
		self.centralPoint = center

		segments = [ComplexArc(center, radius, 0, 2*pi)]
		super(Circle, self).__init__(segments)

	def __str__(self):
		return 'Circle: center={center.real:.3f}{center.imag:+.3f}i, radius={radius:.3f}'.format(center=self.center, radius=self.radius)
	
	def contains(self, z):
		""" Returns True if the point z lies within the contour, False if otherwise """
		return abs(z - self.center) < self.radius

	@property
	def area(self):
		return pi*self.radius**2

	def subdivide(self, axis='r', divisionFactor=0.5):
		"""
		Subdivide the contour

		Parameters
		----------
		axis : str, can only be 'r' (argument kept for consistency with 'subdivisions' method in parent Contour class)
			The axis along which the line subdividing the contour is a constant.
		divisionFactor : float in range (0,1), optional
			Determines the point along 'axis' at which the line dividing the box is placed

		Returns
		-------
		box1 : Annulus
			With inner radius determined by the divisionFactor and outer radius equal to that of the original circle
		box2 : Circle
			With radius equal to the inner radius of box1
		"""
		if axis == 'r' or self.axisName[axis] == 'r':
			box1 = Annulus(self.center, [self.radius*divisionFactor, self.radius])
			box2 = Circle(self.center, self.radius*divisionFactor)
			box1.segments[0] = self.segments[0]
			box1.segments[1]._reversePath = box2.segments[0]
			box2.segments[0]._reversePath = box1.segments[1]

		for box in [box1, box2]:
			box._createdBySubdivisionAxis = axis
			box._parentBox = self
			self._childBoxes = [box1, box2]

		return box1, box2

	def randomPoint(self):
		""" Returns a random point inside the Circle """
		r   = np.random.uniform(0,self.radius)
		phi = np.random.uniform(0,2*pi)
		return r*exp(1j*phi) + self.center

class Annulus(Contour):
	"""
	An annulus in the complex plane with the outer circle positively oriented
	and the inner circle negatively oriented.

	Parameters
	----------
	center : complex
		The center of the annulus in the complex plane.
	radii : list
		A list of length two of the form [inner_radius, outer_radius]
	"""
	def __init__(self, center, radii):
		self.center = center
		self.radii = radii
		self.axisName = ['r', 'phi']

		segments = [ComplexArc(center, radii[1], 0, 2*pi), ComplexArc(center, radii[0], 0, -2*pi)]
		super(Annulus, self).__init__(segments)

	def __str__(self):
		return 'Annulus: center={center.real:.3f}{center.imag:+.3f}i, inner radius={radii[0]:.3f}, outer radius={radii[1]:.3f}'.format(center=self.center, radii=self.radii)
	
	@property
	def centralPoint(self):
		# get the central point within the contour
		r = (self.radii[0] + self.radii[1])/2
		return r

	@property
	def area(self):
		return pi*(self.radii[1]**2 - self.radii[0]**2)

	def contains(self, z):
		""" Returns True if the point z lies within the contour, False if otherwise """
		return self.radii[0] < abs(z - self.center) < self.radii[1]

	def subdivide(self, axis, divisionFactor=0.5):
		"""
		Subdivide the contour

		Parameters
		----------
		axis : str, can be either 'r' or 'phi'
			The axis along which the line subdividing the contour is a constant.
		divisionFactor : float in range (0,1), optional
			Determines the point along 'axis' at which the line dividing the box is placed

		Returns
		-------
		boxes : list of contours
			Two annuluses if axis is 'r'.
			Two half-annuluses oriented according to divisionFactor if axis is 'phi'.
		"""
		if axis == 'r' or self.axisName[axis] == 'r':
			midpoint = self.radii[0] + divisionFactor*(self.radii[1]-self.radii[0])
			box1 = Annulus(self.center, [self.radii[0], midpoint])
			box2 = Annulus(self.center, [midpoint, self.radii[1]])

			box1.segments[1] = self.segments[1]
			box2.segments[0] = self.segments[0]
			box1.segments[0]._reversePath = box2.segments[1]
			box2.segments[1]._reversePath = box1.segments[0]

			box1._createdBySubdivisionAxis = axis
			box2._createdBySubdivisionAxis = axis

		elif axis == 'phi' or self.axisName[axis] == 'phi':
			# Subdividing into two radial boxes rather than one to 
			# ensure that an error is raised if one of the new paths
			# is too close to a root
			# XXX: introduce another parameter for phi1

			phi0 = 2*pi*divisionFactor
			phi1 = phi0 + pi

			box1 = AnnulusSector(self.center, self.radii, [phi0, phi1])
			box2 = AnnulusSector(self.center, self.radii, [phi1, phi0])

			box1.segments[0]._reversePath = box2.segments[2]
			box2.segments[2]._reversePath = box1.segments[0]
			box1.segments[2]._reversePath = box2.segments[0]
			box2.segments[0]._reversePath = box1.segments[2]

		for box in [box1, box2]:
			box._createdBySubdivisionAxis = axis
			box._parentBox = self
		self._childBoxes = [box1, box2]

		return box1, box2

	def randomPoint(self):
		""" Returns a random point inside the Annulus """
		r   = np.random.uniform(*self.radii)
		phi = np.random.uniform(0,2*pi)
		return r*exp(1j*phi) + self.center


class AnnulusSector(Contour):
	"""
	A sector of an annulus in the complex plane.
	
	Parameters
	==========
	center : complex
		The center of the annulus sector.
	rRange : list
		List of length two of the form [inner_radius, outer_radius]
	phiRange : list
		List of length two of the form [phi0, phi1].
		The segment of the contour containing inner and outer circular arcs 
		will be joined, counter clockwise from phi0 to phi1.
	"""
	def __init__(self, center, rRange, phiRange):
		self.center = center
		self.axisName = ['r', 'phi']

		if phiRange[0] > phiRange[1]:
			phiRange[1] += 2*pi

		phi0, phi1 = self.phiRange = phiRange

		# r > 0
		r0, r1 = self.rRange = rRange
		if r0 < 0 or r1 <= 0:
			raise ValueError('Radius > 0')

		# verticies [[radius0,phi0],[radius0,phi1],[radius1,phi1],[radius0,phi1]] 
		self.z1 = z1 = center + r0*exp(1j*phi0)
		self.z2 = z2 = center + r1*exp(1j*phi0)
		self.z3 = z3 = center + r1*exp(1j*phi1)
		self.z4 = z4 = center + r0*exp(1j*phi1)

		segments = [ComplexLine(z1,z2),
					ComplexArc(center,r1,phi0,phi1-phi0),
					ComplexLine(z3,z4),
					ComplexArc(center,r0,phi1,phi0-phi1)]

		super(AnnulusSector, self).__init__(segments)

	def __str__(self):
		return 'Annulus sector: center={center.real:.3f}{center.imag:+.3f}i, r0={rRange[0]:.3f}, r1={rRange[1]:.3f}, phi0={phiRange[0]:.3f}, phi1={phiRange[1]:.3f}'.format(center=self.center, rRange=self.rRange, phiRange=self.phiRange)
	
	@property
	def centralPoint(self):
		# get the central point within the contour
		r = (self.rRange[0] + self.rRange[1])/2
		phi = (self.phiRange[0] + self.phiRange[1])/2
		return r*exp(1j*phi)

	@property
	def area(self):
		return (self.rRange[1]**2 - self.rRange[0]**2)*abs(self.phiRange[1] - self.phiRange[0])%(2*pi)/2

	def contains(self, z):
		""" Returns True if the point z lies within the contour, False if otherwise """
		angle = np.angle(z - self.center)%(2*pi) # np.angle maps to [-pi,pi]
		radiusCorrect = self.rRange[0] < abs(z - self.center) < self.rRange[1]
		
		phi = np.mod(self.phiRange, 2*pi)
		if phi[0] > phi[1]:
			angleCorrect = phi[0] < angle <= 2*pi or 0 <= angle < phi[1]
		else:
			angleCorrect = phi[0] < angle < phi[1]

		return radiusCorrect and angleCorrect

	def subdivide(self, axis, divisionFactor=0.5):
		"""
		Subdivide the contour

		Parameters
		----------
		axis : str, can be either 'r' or 'phi'
			The axis along which the line subdividing the contour is a constant.
		divisionFactor : float in range (0,1), optional
			Determines the point along 'axis' at which the line dividing the box is placed

		Returns
		-------
		box1 : AnnulusSector
			If axis is 'r' then phiRange and the inner radius is the same as original AnnulusSector
			with the outer radius determined by the divisionFactor.
			If axis is 'phi' then the rRange and phiRange[0] is the same as the original AnnulusSector
			with phiRange[1] determined by the divisionFactor.
		box2 : AnnulusSector
			If axis is 'r' then phiRange and the outer radius is the same as original AnnulusSector
			with the inner radius determined equal to the outer radius of box1.
			If axis is 'phi' then the rRange and phiRange[1] is the same as the original AnnulusSector
			with phiRange[0] equal to phiRange[1] of box1.
		"""
		r0, r1 = self.rRange
		phi0, phi1 = self.phiRange
		if axis == 0 or axis == self.axisName[0]:
			divisionPoint = r0 + divisionFactor*(r1-r0)
			box1 = AnnulusSector(self.center, [r0, divisionPoint], self.phiRange)
			box2 = AnnulusSector(self.center, [divisionPoint, r1], self.phiRange)

			# reuse line segments from original box where possible
			# this allows the cached integrals to be used
			box1.segments[3] = self.segments[3]
			box2.segments[1] = self.segments[1]
			box1.segments[1]._reversePath = box2.segments[3]
			box2.segments[3]._reversePath = box1.segments[1]

		elif axis == 1 or axis == self.axisName[1]:
			divisionPoint = phi0 + divisionFactor*(phi1-phi0)
			box1 = AnnulusSector(self.center, self.rRange, [phi0, divisionPoint])
			box2 = AnnulusSector(self.center, self.rRange, [divisionPoint, phi1])

			box1.segments[0] = self.segments[0]
			box2.segments[2] = self.segments[2]
			box1.segments[2]._reversePath = box2.segments[0]
			box2.segments[0]._reversePath = box1.segments[2]

		for box in [box1, box2]:
			box._createdBySubdivisionAxis = axis
			box._parentBox = self
		self._childBoxes = [box1, box2]

		return box1, box2

	def randomPoint(self):
		"""Returns a random point inside the contour of the AnnulusSector."""
		r = np.random.uniform(*self.rRange)
		phiRange = np.mod(self.phiRange, 2*pi)
		if phiRange[0] > phiRange[1]:
			phi = random.choice([np.random.uniform(phiRange[0], 2*pi),
								 np.random.uniform(0, phiRange[1])])
		else:
			phi = np.random.uniform(*phiRange)

		return r*exp(1j*phi) + self.center


class Rectangle(Contour):
	"""
	A positively oriented rectangle in the complex plane.
	
	Parameters
	==========
	xRange : list
		List of length 2 giving the range of the rectangle along the real axis.
	yRange : list
		List of length 2 giving the range of the rectangle along the imaginary axis.
	"""
	def __init__(self, xRange, yRange):
		self.xRange = xRange
		self.yRange = yRange
		self.axisName = ['x', 'y']

		self.z1 = z1 = self.xRange[0] + 1j*self.yRange[0]
		self.z2 = z2 = self.xRange[1] + 1j*self.yRange[0]
		self.z3 = z3 = self.xRange[1] + 1j*self.yRange[1]
		self.z4 = z4 = self.xRange[0] + 1j*self.yRange[1]

		segments = [ComplexLine(z1,z2),
					ComplexLine(z2,z3),
					ComplexLine(z3,z4),
					ComplexLine(z4,z1)]
		super(Rectangle, self).__init__(segments)

	def __str__(self):
		return 'Rectangle: vertices = {z1.real:.3f}{z1.imag:+.3f}i, {z2.real:.3f}{z2.imag:+.3f}i, {z3.real:.3f}{z3.imag:+.3f}i, {z4.real:.3f}{z4.imag:+.3f}i'.format(z1=self.z1, z2=self.z2, z3=self.z3, z4=self.z4)

	@property
	def centralPoint(self):
		# get the central point within the contour
		x = (self.xRange[0] + self.xRange[1])/2
		y = (self.yRange[0] + self.yRange[1])/2
		return x + 1j*y

	@property
	def area(self):
		return (self.xRange[1]-self.xRange[0])*(self.yRange[1]-self.yRange[0])

	def contains(self, z):
		""" Returns True if the point z lies within the contour, False if otherwise """
		return self.xRange[0] < z.real < self.xRange[1] and self.yRange[0] < z.imag < self.yRange[1]

	def subdivide(self, axis, divisionFactor=0.5):
		"""
		Subdivide the contour

		Parameters
		----------
		axis : str, can be either 'x' or 'y'
			The axis along which the line subdividing the contour is a constant.
		divisionFactor : float in range (0,1), optional
			Determines the point along 'axis' at which the line dividing the box is placed

		Returns
		-------
		box1 : Rectangle
			If axis is 'x' then box1 has the same yRange and minimum value of xRange as the 
			original Rectangle but the maximum xRange is determined by the divisionFactor.
			If axis is 'y' then box1 has the same xRange and minimum value of yRange as the
			original Rectangle but the maximum yRange is determined by the divisionFactor.
		box2 : Rectangle
			If axis is 'x' then box2 has the same yRange and maximum value of xRange as the 
			original Rectangle but the minimum xRange is equal to the maximum xRange of box1.
			If axis is 'x' then box2 has the same xRange and maximum value of yRange as the 
			original Rectangle but the minimum yRange is equal to the maximum yRange of box1.
		"""
		if axis == 'x' or self.axisName[axis] == 'x':
			midpoint = self.xRange[0] + divisionFactor*(self.xRange[1]-self.xRange[0])
			box1 = Rectangle([self.xRange[0], midpoint], self.yRange)
			box2 = Rectangle([midpoint, self.xRange[1]], self.yRange)

			box1.segments[3] = self.segments[3]
			box2.segments[1] = self.segments[1]
			box1.segments[1]._reversePath = box2.segments[3]
			box2.segments[3]._reversePath = box1.segments[1]

		elif axis == 'y' or self.axisName[axis] == 'y':
			midpoint = self.yRange[0] + divisionFactor*(self.yRange[1]-self.yRange[0])
			box1 = Rectangle(self.xRange, [self.yRange[0], midpoint])
			box2 = Rectangle(self.xRange, [midpoint, self.yRange[1]])

			box1.segments[0] = self.segments[0]
			box2.segments[2] = self.segments[2]
			box1.segments[2]._reversePath = box2.segments[0]
			box2.segments[0]._reversePath = box1.segments[2]

		for box in [box1, box2]:
			box._createdBySubdivisionAxis = axis
			box._parentBox = self
		self._childBoxes = [box1, box2]

		return box1, box2

	def randomPoint(self):
		"""Returns a random point inside the contour of the Rectangle."""
		x = np.random.uniform(*self.xRange)
		y = np.random.uniform(*self.yRange)
		return x + 1j*y


def divisionFactorGen():
	"""A generator for divisionFactors"""
	x = 0.5
	yield x
	for power in [1e1, 1e2, 1e3]:
		for diff in np.linspace(0, 0.5, int(1+power/2))[1:-1]:
			yield x + diff
			yield x - diff
