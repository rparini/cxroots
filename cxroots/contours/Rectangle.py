from __future__ import division
import numpy as np

from ..Contour import Contour
from ..Paths import ComplexLine

class Rectangle(Contour):
	"""
	A positively oriented rectangle in the complex plane.

	Parameters
	----------
	xRange : tuple
		Tuple of length two giving the range of the rectangle along the
		real axis.
	yRange : tuple
		Tuple of length two giving the range of the rectangle along the
		imaginary axis.

	Examples
	--------
	.. plot::
		:include-source:

		from cxroots import Rectangle
		rect = Rectangle(xRange=(-2, 2), yRange=(-1, 1))
		rect.show()
	"""
	def __init__(self, xRange, yRange):
		self.xRange = xRange
		self.yRange = yRange
		self.axisName = ('x', 'y')

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
			The axis along which the line subdividing the contour is a
			constant.
		divisionFactor : float in range (0,1), optional
			Determines the point along 'axis' at which the line dividing
			the contour is placed.

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
