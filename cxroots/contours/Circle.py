from __future__ import division
import numpy as np
from numpy import pi

from ..Contour import Contour
from ..Paths import ComplexArc
from .Annulus import Annulus

class Circle(Contour):
	"""
	A positively oriented circle in the complex plane.

	Parameters
	----------
	center : complex
		The center of the circle.
	radius : float
		The radius of the circle.

	Examples
	--------
	.. plot::
		:include-source:

		from cxroots import Circle
		circle = Circle(center=1, radius=0.5)
		circle.show()
	"""
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius
		self.axisName = ('r')

		segments = [ComplexArc(center, radius, 0, 2*pi)]
		super(Circle, self).__init__(segments)

	def __str__(self):
		return 'Circle: center={center.real:.3f}{center.imag:+.3f}i, radius={radius:.3f}'.format(center=self.center, radius=self.radius)

	def contains(self, z):
		""" Returns True if the point z lies within the contour, False if otherwise """
		return abs(z - self.center) < self.radius

	@property
	def centralPoint(self):
		return self.center

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
