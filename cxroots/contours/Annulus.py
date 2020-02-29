from __future__ import division
import numpy as np
from numpy import pi

from ..Contour import Contour
from ..Paths import ComplexArc
from .AnnulusSector import AnnulusSector

class Annulus(Contour):
	"""
	An annulus in the complex plane with the outer circle positively oriented
	and the inner circle negatively oriented.

	Parameters
	----------
	center : complex
		The center of the annulus in the complex plane.
	radii : tuple
		A tuple of length two of the form (inner_radius, outer_radius).

	Examples
	--------
	.. plot::
		:include-source:

		from cxroots import Annulus
		annulus = Annulus(center=0, radii=(0.5,0.75))
		annulus.show()
	"""
	def __init__(self, center, radii):
		self.center = center
		self.radii = radii
		self.axisName = ('r', 'phi')

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
