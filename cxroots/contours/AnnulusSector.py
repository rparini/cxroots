from __future__ import division
import numpy as np
from numpy import pi, exp

from ..Contour import Contour
from ..Paths import ComplexLine, ComplexArc

class AnnulusSector(Contour):
	"""
	A sector of an annulus in the complex plane.

	Parameters
	----------
	center : complex
		The center of the annulus sector.
	radii : tuple
		Tuple of length two of the form (inner_radius, outer_radius)
	phiRange : tuple
		Tuple of length two of the form (phi0, phi1).
		The segment of the contour containing inner and outer circular
		arcs will be joined, counter clockwise from phi0 to phi1.

	Examples
	--------
	.. plot::
		:include-source:

		from numpy import pi
		from cxroots import AnnulusSector
		annulusSector = AnnulusSector(center=0.2, radii=(0.5, 1.25), phiRange=(-pi/4, pi/4))
		annulusSector.show()

	.. plot::
		:include-source:

		from numpy import pi
		from cxroots import AnnulusSector
		annulusSector = AnnulusSector(center=0.2, radii=(0.5, 1.25), phiRange=(pi/4, -pi/4))
		annulusSector.show()
	"""
	def __init__(self, center, radii, phiRange):
		self.center = center
		self.axisName = ('r', 'phi')

		if phiRange[0] > phiRange[1]:
			phiRange = (phiRange[0], phiRange[1]+2*pi)

		phi0, phi1 = self.phiRange = phiRange

		# r > 0
		r0, r1 = self.radii = radii
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
		return 'Annulus sector: center={center.real:.3f}{center.imag:+.3f}i, r0={radii[0]:.3f}, r1={radii[1]:.3f}, phi0={phiRange[0]:.3f}, phi1={phiRange[1]:.3f}'.format(center=self.center, radii=self.radii, phiRange=self.phiRange)

	@property
	def centralPoint(self):
		# get the central point within the contour
		r = (self.radii[0] + self.radii[1])/2
		phi = (self.phiRange[0] + self.phiRange[1])/2
		return r*exp(1j*phi)

	@property
	def area(self):
		return (self.radii[1]**2 - self.radii[0]**2)*abs(self.phiRange[1] - self.phiRange[0])%(2*pi)/2

	def contains(self, z):
		""" Returns True if the point z lies within the contour, False if otherwise """
		angle = np.angle(z - self.center)%(2*pi) # np.angle maps to [-pi,pi]
		radiusCorrect = self.radii[0] < abs(z - self.center) < self.radii[1]

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
			If axis is 'phi' then the radii and phiRange[0] is the same as the original AnnulusSector
			with phiRange[1] determined by the divisionFactor.
		box2 : AnnulusSector
			If axis is 'r' then phiRange and the outer radius is the same as original AnnulusSector
			with the inner radius determined equal to the outer radius of box1.
			If axis is 'phi' then the radii and phiRange[1] is the same as the original AnnulusSector
			with phiRange[0] equal to phiRange[1] of box1.
		"""
		r0, r1 = self.radii
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
			box1 = AnnulusSector(self.center, self.radii, [phi0, divisionPoint])
			box2 = AnnulusSector(self.center, self.radii, [divisionPoint, phi1])

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
		r = np.random.uniform(*self.radii)
		phiRange = np.mod(self.phiRange, 2*pi)
		if phiRange[0] > phiRange[1]:
			phi = random.choice([np.random.uniform(phiRange[0], 2*pi),
								 np.random.uniform(0, phiRange[1])])
		else:
			phi = np.random.uniform(*phiRange)

		return r*exp(1j*phi) + self.center
