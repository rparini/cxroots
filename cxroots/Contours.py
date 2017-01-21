"""
References
----------
[DSZ] "Locating all the Zeros of an Analytic Function in one Complex Variable"
	M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2
[BVP] Gakhov, F. D. "Boundary value problems", section 12 (2014), Elsevier.
[DL] "A Numerical Method for Locating the Zeros of an Analytic function", 
	L.M.Delves, J.N.Lyness, Mathematics of Computation (1967), Vol.21, Issue 100
"""

from __future__ import division
import random
import warnings
import numpy as np
import scipy.integrate
from scipy import pi, exp, sin, log
import scipy

from .ContArg import contArg

class ComplexPath(object):
	""" A base class for paths in the complex plane """
	def __init__(self):
		self._integralCache = {}
		self._contArgCache = {}

	def plot(self, N=100, linecolor='b', linestyle='-'):
		"""
		Use matplotlib to plot, but not show, the path as a 
		2D plot in the Complex plane.  To show use pyplot.show()

		Parameters
		----------
		N : int, optional
			The number of points to use when plotting the path.  Default is 100
		linecolor : optional
			The colour of the plotted path, passed to the pyplot.plot function 
			as the keyword argument of 'color'.
		linestyle : str, optional
			The line style of the plotted path, passed to the pyplot.plot 
			function as the keyword argument of 'linestyle'.  Default is '-' 
			which corresponds to a solid line.
		"""
		import matplotlib.pyplot as plt
		t = np.linspace(0,1,N)
		path = self(t)
		plt.plot(path.real, path.imag, color=linecolor, linestyle=linestyle)
		plt.xlabel('Re[$z$]', size=16)
		plt.ylabel('Im[$z$]', size=16)

		# add arrow to indicate direction of path
		arrow_direction = (self(0.51) - self(0.5))/abs(self(0.51) - self(0.5))
		arrow_extent = 1e-6*arrow_direction
		ymin, ymax = plt.gca().get_ylim()
		xmin, xmax = plt.gca().get_xlim()
		head_length = max(abs(ymax - ymin), abs(xmax - xmin))/40.
		plt.arrow(self(0.5).real, self(0.5).imag,
				  arrow_extent.real, arrow_extent.imag,
				  head_width=head_length*2/3., head_length=head_length, 
				  fc=linecolor, ec=linecolor)

	def show(self, *args, **kwargs):
		""" Shows the path as a 2D plot in the complex plane using 
		the same arguments as the plot method """
		import matplotlib.pyplot as plt
		self.plot(*args, **kwargs)
		plt.show()

	def integrate(self, f, tol=1e-8, rombergDivMax=10, show=False):
		"""
		Integrate the function f along the path using SciPy's Romberg
		algorithm.  The value of the integral is cached and will be
		reused if the method is called with same function f and tol. 

		Parameters
		----------
		f : function of a single complex variable
		tol : float, optional
			The absolute tolerance passed to SciPy's Romberg function.
			Default is 1e-8.
		rombergDivMax : int, optional
			The maximum order of extrapolation passed to SciPy's Romberg function

		Returns
		-------
		integral : complex
			The integral of the function f along the path.
		"""

		args = (f, tol)
		if args in self._integralCache.keys():
			integral = self._integralCache[args]

		elif hasattr(self, '_reversePath') and args in self._reversePath._integralCache:
			# if we have already computed the reverse of this path
			integral = -self._reversePath._integralCache[args]

		else:			
			# suppress accuracy warnings
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				integrand = lambda t: f(self(t))*self.dzdt(t)
				integral = scipy.integrate.romberg(integrand,0,1,tol=tol,divmax=rombergDivMax,show=show)

			if np.isnan(integral):
				raise RuntimeError('The integral along the segment %s is NaN.\
					\nThis is most likely due to a root being on or very close to the path of integration.'%self)

			self._integralCache[args] = integral

		return integral


class ComplexLine(ComplexPath):
	""" A straight line in the complex plane from a to b """
	def __init__(self, a, b):
		self.a, self.b = a, b
		self.dzdt = lambda t: self.b-self.a
		super(ComplexLine, self).__init__()

	def __str__(self):
		return 'ComplexLine from %.3f+%.3fi to %.3f+%.3fi' % (self.a.real, self.a.imag, self.b.real, self.b.imag)

	def __call__(self, t):
		""" The parameterization of the line in the variable t, where 0 <= t <= 1 """
		return self.a + t*(self.b-self.a)

class ComplexArc(ComplexPath):
	""" An arc with center z0, radius R, initial angle t0 and change of angle dt """
	def __init__(self, z0, R, t0, dt):
		self.z0, self.R, self.t0, self.dt = z0, R, t0, dt
		self.dzdt = lambda t: 1j*self.dt*self.R*exp(1j*(self.t0 + t*self.dt))
		super(ComplexArc, self).__init__()

	def __str__(self):
		return 'ComplexArc: z0=%.3f, R=%.3f, t0=%.3f, dt=%.3f' % (self.z0, self.R, self.t0, self.dt)

	def __call__(self, t):
		""" The parameterization of the arc in the variable t, where 0 <= t <= 1 """
		return self.R*exp(1j*(self.t0 + t*self.dt)) + self.z0

class NonIntegerError(Exception):
    pass


# create a cache of the integrands created in the enclosed_zeros method
# of the PolarRect class so that the segment integrals can be cached
zerosIntegrandCache = {}

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

	def show(self, *args, **kwargs):
		""" Shows the path as a 2D plot in the complex plane using 
		the same arguments as the plot method """
		import matplotlib.pyplot as plt
		self.sizePlot()
		self.plot(*args, **kwargs)
		plt.show()

	def integrate(self, f, tol=1e-8, rombergDivMax=10, show=False):
		""" Integrate around the contour, same arguments the integrate method for ComplexPath
		but the tolerance passed to each segment will be tol/len(self.segments) """
		segmentTol = tol/len(self.segments)
		return sum([segment.integrate(f, segmentTol, rombergDivMax, show) for segment in self.segments])

	def enclosed_zeros(self, f, df=None, integerTol=0.1, rombergDivMax=10, reqEqualZeros=3):
		r"""
		For a function of one complex variable, f(z), which is analytic in and within the contour,
		return the number of zeros (including multiplicities) within the contour calculated, using 
		Cauchy's argument principle, as
		
		.. math::

			\frac{1}{2i\pi} \oint_C \frac{f'(z)}{f(z)} dz.

		If df(z), the derivative of f(z), is provided then the above integral is computed directly
		using `SciPy's implementation of Romberg integration <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romberg.html>`_.

		If only f(z) is provided then instead the number of zeros is computed as the difference
		of the argument of f(z) continued around the contour [DL,BVP].  This has the potential to be
		unreliable since this method will always find that the integral is an integer, rather than
		smoothly converging to an integer as the number of points at which f is sampled increases.  This
		makes it difficult to tell when the number of zeros has genuinely converged.  
		In this case the number of sample points is doubled until the last reqEqualZeros (3 by default) evaluations of 
		the number of zeros are equal and non-negative.  The contour is also rejected as being 
		unreliable if at any point :math:`|df/f| > 10^6`.
		
		Parameters
		----------
		f : function
			Function of a single variable f(x)
		df : function, optional
			Function of a single variable, df(x), providing the
			derivative of the function f(x) at the point x
		integerTol : float, optional
			How close the result of the Romberg integration has to be to an integer for it to be
			accepted (only used if df is given).  The absolute tolerance of the Romberg integration
			will be integerTol/2.
		rombergDivMax : int, optional
			The maximum order of extrapolation of the Romberg integration routine (only used if df is given)
		reqEqualZeros : int, optional
			If the Cauchy integral is computed by continuing the argument around the contour (ie. if df is None)
			then the routine requires that the last reqEqualZeros evaluations of the number of enclosed zeros 
			are equal and non-negative.  Default is 3.

		Returns
		-------
		int
			The number of zeros of f (including multiplicities) which lie within the contour
		
		References
		----------
		[DSZ] "Locating all the Zeros of an Analytic Function in one Complex Variable" 
			M.Dellnitz, O.Schutze, Q.Zheng, J. Compu. and App. Math. (2002), Vol.138, Issue 2

		[BVP] Gakhov, F. D. "Boundary value problems", section 12 (2014), Elsevier.
		"""
		if df is None:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")

				N = 50 # initial number of sample points

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

					df = np.diff(val)/np.diff(z)

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

		else:
			# reuse the same integrand so that the same integrals along individual paths can be reused
			if (f, df) in zerosIntegrandCache.keys():
				integrand = zerosIntegrandCache[(f, df)]
			else:
				integrand = lambda z: df(z)/(2j*pi*f(z))
				zerosIntegrandCache[(f,df)] = integrand

			integralTol = integerTol/2.
			integral = self.integrate(integrand, tol=integralTol, rombergDivMax=rombergDivMax)
			numberOfZeros = int(round(integral.real))
			if numberOfZeros < 0 or abs(integral.real - numberOfZeros) > integerTol or abs(integral.imag) > integerTol:
				raise NonIntegerError('Value of contour integral %s is not sufficiently close to a positive integer'%integral)

		return numberOfZeros

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


class Circle(Contour):
	"""A positively oriented circle."""
	def __init__(self, center, radius):
		self.center = self.centerPoint = center
		self.radius = radius
		self.axisName = ['r']
		self.centerPoint = center

		segments = [ComplexArc(center, radius, 0, 2*pi)]
		super(Circle, self).__init__(segments)

	def __str__(self):
		return 'Circle: center=%.3f, radius=%.3f' % (self.center, self.radius)
	
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
			return box1, box2

	def randomPoint(self):
		""" Returns a random point inside the Circle """
		r   = np.random.uniform(0,self.radius)
		phi = np.random.uniform(0,2*pi)
		return r*exp(1j*phi) + self.center

class Annulus(Contour):
	"""
	An annulus with given center and radii=[inner_radius, outer_radius].
	The outer circle is positively oriented and the inner circle is
	negatively oriented.
	"""
	def __init__(self, center, radii):
		self.center = center
		self.radii = radii
		self.axisName = ['r', 'phi']

		segments = [ComplexArc(center, radii[1], 0, 2*pi), ComplexArc(center, radii[0], 0, -2*pi)]
		super(Annulus, self).__init__(segments)

	def __str__(self):
		return 'Annulus: center=%.3f, inner radius=%.3f, outer radius=%.3f' % (self.center, self.radii[0], self.radii[1])
	
	@property
	def centerPoint(self):
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
			boxes = [box1, box2]

		elif axis == 'phi' or self.axisName[axis] == 'phi':
			# Subdividing into two radial boxes rather than one to 
			# ensure that an error is raised if one of the new paths
			# is too close to a root
			# XXX: introduce another parameter for phi1

			phi0 = 2*pi*divisionFactor
			phi1 = phi0 + pi

			box1 = PolarRect(self.center, self.radii, [phi0, phi1])
			box2 = PolarRect(self.center, self.radii, [phi1, phi0])

			box1.segments[0]._reversePath = box2.segments[2]
			box2.segments[2]._reversePath = box1.segments[0]
			box1.segments[2]._reversePath = box2.segments[0]
			box2.segments[0]._reversePath = box1.segments[2]
			boxes = [box1, box2]

		for box in boxes:
			box._createdBySubdivisionAxis = axis
		return boxes

	def randomPoint(self):
		""" Returns a random point inside the Annulus """
		r   = np.random.uniform(*self.radii)
		phi = np.random.uniform(0,2*pi)
		return r*exp(1j*phi) + self.center


class PolarRect(Contour):
	"""
	A positively oriented contour which is a rectangle in polar 
	coordinates with verticies: [[radius0,phi0],[radius0,phi1],[radius1,phi1],[radius0,phi1]] 
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

		self.z1 = z1 = center + r0*exp(1j*phi0)
		self.z2 = z2 = center + r1*exp(1j*phi0)
		self.z3 = z3 = center + r1*exp(1j*phi1)
		self.z4 = z4 = center + r0*exp(1j*phi1)

		segments = [ComplexLine(z1,z2),
					ComplexArc(center,r1,phi0,phi1-phi0),
					ComplexLine(z3,z4),
					ComplexArc(center,r0,phi1,phi0-phi1)]

		super(PolarRect, self).__init__(segments)

	def __str__(self):
		return 'Polar rectangle: r0=%.3f, r1=%.3f, phi0=%.3f, phi1=%.3f' % (self.rRange[0], self.rRange[1], self.phiRange[0], self.phiRange[1])
	
	@property
	def centerPoint(self):
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
			angleCorrect = phi[0] < angle < 2*pi or 0 < angle < phi[1]
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
		box1 : PolarRect
			If axis is 'r' then phiRange and the inner radius is the same as original PolarRect
			with the outer radius determined by the divisionFactor.
			If axis is 'phi' then the rRange and phiRange[0] is the same as the original PolarRect
			with phiRange[1] determined by the divisionFactor.
		box2 : PolarRect
			If axis is 'r' then phiRange and the outer radius is the same as original PolarRect
			with the inner radius determined equal to the outer radius of box1.
			If axis is 'phi' then the rRange and phiRange[1] is the same as the original PolarRect
			with phiRange[0] equal to phiRange[1] of box1.
		"""
		r0, r1 = self.rRange
		phi0, phi1 = self.phiRange
		if axis == 0 or axis == self.axisName[0]:
			divisionPoint = r0 + divisionFactor*(r1-r0)
			box1 = PolarRect(self.center, [r0, divisionPoint], self.phiRange)
			box2 = PolarRect(self.center, [divisionPoint, r1], self.phiRange)

			# reuse line segments from original box where possible
			# this allows the cached integrals to be used
			box1.segments[3] = self.segments[3]
			box2.segments[1] = self.segments[1]
			box1.segments[1]._reversePath = box2.segments[3]
			box2.segments[3]._reversePath = box1.segments[1]

		elif axis == 1 or axis == self.axisName[1]:
			divisionPoint = phi0 + divisionFactor*(phi1-phi0)
			box1 = PolarRect(self.center, self.rRange, [phi0, divisionPoint])
			box2 = PolarRect(self.center, self.rRange, [divisionPoint, phi1])

			box1.segments[0] = self.segments[0]
			box2.segments[2] = self.segments[2]
			box1.segments[2]._reversePath = box2.segments[0]
			box2.segments[0]._reversePath = box1.segments[2]

		for box in [box1, box2]:
			box._createdBySubdivisionAxis = axis
		return box1, box2

	def randomPoint(self):
		"""Returns a random point inside the contour of the PolarRect."""
		r = np.random.uniform(*self.rRange)
		phiRange = np.mod(self.phiRange, 2*pi)
		if phiRange[0] > phiRange[1]:
			phi = random.choice([np.random.uniform(phiRange[0], 2*pi),
								 np.random.uniform(0, phiRange[1])])
		else:
			phi = np.random.uniform(*phiRange)

		return r*exp(1j*phi) + self.center


class Rectangle(Contour):
	"""A positively oriented rectangle in the complex plane"""
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
		return "Rectangle: %.3f+i%.3f, %.3f+i%.3f, %.3f+i%.3f, %.3f+i%.3f"%(self.z1.real, self.z1.imag, self.z2.real, self.z2.imag, self.z3.real, self.z3.imag, self.z4.real, self.z4.imag)

	@property
	def centerPoint(self):
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
		for diff in np.linspace(0, 0.5, 1+power/2)[1:-1]:
			yield x + diff
			yield x - diff
