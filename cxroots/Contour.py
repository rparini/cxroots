from __future__ import division

import numpy as np
import functools

from .CountRoots import count_roots
from .ApproximateRoots import approximate_roots
from .RootFinder import find_roots
from .DemoRootFinder import demo_find_roots
from .Misc import remove_para
from .Paths import ComplexPath


class Contour(object):
	"""
	A base class for contours in the complex plane.

	Attributes
	----------
	centralPoint : complex
		The point at the center of the contour.
	area : float
		The surface area of the contour.
	"""
	def __init__(self, segments):
		self.segments = np.array(segments, dtype=object)

	def __call__(self, t):
		r"""
		The point on the contour corresponding the value of the
		parameter t.

		Parameters
		----------
		t : float
			A real number :math:`0\leq t \leq 1` which parameterises
			the contour.

		Returns
		-------
		complex
			A point on the contour.

		Example
		-------

		>>> from cxroots.Paths import Circle
		>>> c = Circle(0,1)	# Circle |z|=1 parameterised by e^{it}
		>>> c(0.25)
		(6.123233995736766e-17+1j)
		>>> c(0) == c(1)
		True
		"""
		t = np.array(t)
		N = len(self.segments)
		segmentIndex = np.array(N*t, dtype=int)
		segmentIndex = np.mod(segmentIndex, N)

		if hasattr(segmentIndex, '__iter__'):
			return np.array([self.segments[i](N*t[ti]%1) for ti, i in enumerate(segmentIndex)])
		else:
			return self.segments[segmentIndex](N*t%1)

	@property
	def centralPoint(self):
		raise NotImplementedError('centralPoint needs to be implemented in the subclass.')

	@property
	def area(self):
		raise NotImplementedError('area needs to be implemented in the subclass.')

	def contains(self, z):
		"""
		Tests whether the point z is within the contour.

		Parameters
		----------
		z : complex

		Returns
		-------
		bool
			True if z lies within the contour and false otherwise.
		"""
		raise NotImplementedError('contains() needs to be implemented in the subclass.')

	@functools.wraps(ComplexPath.plot)
	def plot(self, *args, **kwargs):
		self._sizePlot()
		for segment in self.segments:
			segment.plot(*args, **kwargs)

	def _sizePlot(self):
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

	def show(self, saveFile=None, **plotKwargs):
		"""
		Shows the contour as a 2D plot in the complex plane.  Requires
		Matplotlib.

		Parameters
		----------
		saveFile : str (optional)
			If given then the plot will be saved to disk with name
			'saveFile'.  If saveFile=None the plot is shown on-screen.
		**plotKwargs
			Key word arguments are as in :meth:`~cxroots.Contour.Contour.plot`.
		"""
		import matplotlib.pyplot as plt
		self.plot(**plotKwargs)

		if saveFile is not None:
			plt.savefig(saveFile, bbox_inches='tight')
			plt.close()
		else:
			plt.show()

	def subdivisions(self, axis='alternating'):
		"""
		A generator for possible subdivisions of the contour.

		Parameters
		----------
		axis : str, 'alternating' or any element of self.axisName.
			The axis along which the line subdividing the contour is a
			constant (eg. subdividing a circle along the radial axis
			will give an outer annulus and an inner circle).  If
			alternating then the dividing axis will always be different
			to the dividing axis used to create the contour which is now
			being divided.

		Yields
		------
		tuple
			A tuple with two contours which subdivide the original
			contour.
		"""
		if axis == 'alternating':
			if hasattr(self,'_createdBySubdivisionAxis'):
				axis = (self._createdBySubdivisionAxis + 1)%len(self.axisName)
			else:
				axis = 0

		for divisionFactor in divisionFactorGen():
			yield self.subdivide(axis, divisionFactor)

	def distance(self, z):
		"""
		Get the distance from the point z in the complex plane to the
		nearest point on the contour.

		Parameters
		----------
		z : complex
			The point from which to measure the distance to the closest
			point on the contour to z.

		Returns
		-------
		float
			The distance from z to the point on the contour which is
			closest to z.
		"""
		return min(segment.distance(z) for segment in self.segments)

	@functools.wraps(ComplexPath.integrate)
	def integrate(self, f, **integrationKwargs):
		return sum([segment.integrate(f, **integrationKwargs) for segment in self.segments])

	@remove_para('C')
	@functools.wraps(count_roots)
	def count_roots(self, f, df=None, **kwargs):
		return count_roots(self, f, df, **kwargs)

	@remove_para('C')
	@functools.wraps(approximate_roots)
	def approximate_roots(self, N, f, df=None, **kwargs):
		return approximate_roots(self, N, f, df, **kwargs)

	@remove_para('originalContour')
	@functools.wraps(find_roots)
	def roots(self, f, df=None, **kwargs):
		return find_roots(self, f, df, **kwargs)

	@remove_para('C')
	@functools.wraps(demo_find_roots)
	def demo_roots(self, *args, **kwargs):
		return demo_find_roots(self, *args, **kwargs)

def divisionFactorGen():
	"""A generator for divisionFactors."""
	yield 0.3	# being off-center is a better first choice for certain problems

	x = 0.5
	yield x
	for diff in np.linspace(0, 0.5, int(1+10/2.))[1:-1]:
		yield x + diff
		yield x - diff
